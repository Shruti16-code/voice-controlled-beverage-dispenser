"""
Voice-Controlled Automated Beverage Dispenser
===============================================

A speech-controlled beverage dispensing assistant. The user describes a
drink out loud, a language model converts that request into a recipe
using only the ingredients physically connected to the dispenser, and
the recipe is sent to a microcontroller over a serial connection to
dispense the drink.

Hardware: STM32-based dispenser board (4 ingredient pump channels)
Voice I/O: Google Speech Recognition (input), gTTS (output)
Recipe generation: LLM via OpenRouter API

Environment variables required (see .env.example):
    OPENROUTER_API_KEY - API key for OpenRouter
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import scrolledtext
from typing import Optional

import openai
import serial
import serial.tools.list_ports
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from playsound import playsound

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dispenser")

LLM_MODEL = "google/gemma-2-9b-it"
SERIAL_BAUD_RATE = 115200
SERIAL_TIMEOUT_SECONDS = 25
DISPENSER_PINS = [4, 5, 6, 7]
DEFAULT_CALIBRATION_ML_PER_UNIT = 15.0
LISTEN_TIMEOUT_SECONDS = 5
PHRASE_TIME_LIMIT_SECONDS = 10
TRIGGER_PHRASES = ("make a drink", "mix a drink", "get me a drink", "make me a drink")
EXIT_PHRASES = ("exit", "goodbye")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant for an automated beverage dispenser. "
    "Keep responses brief and friendly."
)


class ConversationState:
    """Tracks what the assistant is currently expecting from the user."""

    IDLE = "IDLE"
    AWAITING_DRINK_DESCRIPTION = "AWAITING_DRINK_DESCRIPTION"


@dataclass
class Recipe:
    """A drink recipe expressed as ingredient -> millilitres."""

    ingredients_ml: dict[str, float] = field(default_factory=dict)
    title: str = "Untitled Creation"

    def is_empty(self) -> bool:
        return not self.ingredients_ml


# ------------------------------------------------------------------------
# Language model client
# ------------------------------------------------------------------------

class RecipeGenerator:
    """Generates drink recipes and names from natural-language requests
    using an LLM, constrained to the ingredients actually available."""

    def __init__(self, api_key: str, model: str = LLM_MODEL) -> None:
        if not api_key:
            raise ValueError("An OpenRouter API key is required.")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = model

    def generate_recipe(self, user_request: str, available_ingredients: list[str]) -> Recipe:
        """Ask the LLM for a JSON recipe using only available ingredients."""
        prompt = (
            f"User request: '{user_request}'. "
            f"Available ingredients: {', '.join(available_ingredients)}. "
            "Create a recipe using ONLY these ingredients. Total volume "
            "should be approximately 150-200ml. Respond ONLY with a valid "
            'JSON object mapping ingredient name to millilitres, e.g. '
            '{"vodka": 100, "water": 50}. If no reasonable recipe is '
            "possible, return an empty JSON object: {}."
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.choices[0].message.content
            logger.debug("LLM raw recipe response: %s", raw_text)
            cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

            if not cleaned.startswith("{"):
                logger.warning("LLM returned a non-JSON response for recipe request.")
                return Recipe()

            ingredients_ml = json.loads(cleaned)
            return Recipe(ingredients_ml=ingredients_ml)
        except Exception:
            logger.exception("Recipe generation failed.")
            return Recipe()

    def generate_title(self, recipe: Recipe) -> str:
        """Ask the LLM to invent a short name for the given recipe."""
        prompt = (
            f"Given this recipe: {json.dumps(recipe.ingredients_ml)}. "
            "Invent a short, catchy name for it. Respond ONLY with the "
            'name in quotes, e.g. "Electric Sunset".'
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip().strip('"')
        except Exception:
            logger.exception("Title generation failed.")
            return "Untitled Creation"

    def chat(self, user_message: str) -> str:
        """General-purpose conversational fallback response."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logger.exception("Chat completion failed.")
            return "I'm having trouble thinking right now. Please try again in a moment."


# ------------------------------------------------------------------------
# Voice I/O
# ------------------------------------------------------------------------

class VoiceInterface:
    """Handles text-to-speech output and speech-to-text input."""

    def __init__(self, on_status_update=None) -> None:
        self._recognizer = sr.Recognizer()
        self._is_speaking = False
        self._on_status_update = on_status_update or (lambda _text: None)

    def speak(self, text: str) -> None:
        """Synthesize and play spoken audio for the given text."""
        self._is_speaking = True
        logger.info("Assistant: %s", text)
        self._on_status_update(f"Assistant: {text}")

        filename = f"voice_{random.randint(1000, 9999)}.mp3"
        try:
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            playsound(filename)
        except Exception:
            logger.exception("Text-to-speech playback failed.")
        finally:
            if os.path.exists(filename):
                os.remove(filename)
            self._is_speaking = False

    def listen(self) -> Optional[str]:
        """Capture one spoken command from the microphone and transcribe it.

        Returns the lowercase transcript, or None if nothing was understood.
        """
        if self._is_speaking:
            return None

        with sr.Microphone() as source:
            self._on_status_update("Listening...")
            self._recognizer.pause_threshold = 1.5
            self._recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self._recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT_SECONDS,
                    phrase_time_limit=PHRASE_TIME_LIMIT_SECONDS,
                )
                self._on_status_update("Processing...")
                command = self._recognizer.recognize_google(audio, language="en-in")
                logger.info("User: %s", command)
                self._on_status_update(f"You: {command}")
                return command.lower()
            except sr.WaitTimeoutError:
                self._on_status_update("No speech detected.")
            except sr.UnknownValueError:
                self._on_status_update("Didn't catch that.")
            except Exception:
                logger.exception("Speech recognition failed.")
                self._on_status_update("Speech recognition error.")
        return None


# ------------------------------------------------------------------------
# Dispenser hardware control
# ------------------------------------------------------------------------

class DispenserHardware:
    """Manages the serial connection and pump calibration for the
    physical dispenser board."""

    def __init__(self, port: str, baud_rate: int = SERIAL_BAUD_RATE) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.pin_map: dict[str, int] = {}
        self.calibration_ml_per_unit: dict[str, float] = {}

    @staticmethod
    def find_serial_port() -> Optional[str]:
        """Auto-detect the dispenser board's serial port."""
        logger.info("Searching for dispenser board...")
        for port in serial.tools.list_ports.comports():
            if "STMicroelectronics" in port.description or "STLink" in port.description:
                logger.info("Found dispenser board on %s", port.device)
                return port.device
        return None

    def configure_ingredients(self, ingredient_by_pin: dict[int, str]) -> None:
        """Register which ingredient is loaded on each pump pin, using a
        default calibration value for each."""
        for pin, ingredient_name in ingredient_by_pin.items():
            name = ingredient_name.strip().lower()
            if not name:
                continue
            self.pin_map[name] = pin
            self.calibration_ml_per_unit[name] = DEFAULT_CALIBRATION_ML_PER_UNIT
            logger.info("Pin %d configured with ingredient: %s", pin, name.capitalize())

    def build_dispense_command(self, recipe: Recipe) -> str:
        """Translate a recipe into the pin/duration command string
        understood by the dispenser firmware."""
        parts = []
        for ingredient, millilitres in recipe.ingredients_ml.items():
            if ingredient not in self.pin_map:
                continue
            pin = self.pin_map[ingredient]
            calibration = self.calibration_ml_per_unit[ingredient]
            duration_units = int((millilitres / calibration) * 1000)
            parts.append(f"{pin},{duration_units}")
        return ",".join(parts) + ",0\n" if parts else ""

    def dispense(self, command: str) -> str:
        """Send a dispense command over serial and return the board's
        confirmation response."""
        with serial.Serial(self.port, self.baud_rate, timeout=SERIAL_TIMEOUT_SECONDS) as connection:
            connection.write(command.encode("utf-8"))
            return connection.readline().decode("utf-8").strip()


def prompt_for_bar_setup() -> dict[int, str]:
    """Interactively ask which ingredient is loaded on each available pin."""
    print("--- Bar Setup ---")
    ingredient_by_pin: dict[int, str] = {}
    for pin in DISPENSER_PINS:
        ingredient_name = input(f"What ingredient is on Pin {pin}? (or 'skip'): ").strip().lower()
        if ingredient_name and ingredient_name != "skip":
            ingredient_by_pin[pin] = ingredient_name
    return ingredient_by_pin


# ------------------------------------------------------------------------
# Command routing / application logic
# ------------------------------------------------------------------------

class DispenserAssistant:
    """Coordinates voice I/O, recipe generation, and dispenser hardware
    to fulfill spoken drink requests."""

    def __init__(
        self,
        voice: VoiceInterface,
        recipes: RecipeGenerator,
        hardware: DispenserHardware,
        on_exit=None,
    ) -> None:
        self.voice = voice
        self.recipes = recipes
        self.hardware = hardware
        self.on_exit = on_exit or (lambda: None)
        self.state = ConversationState.IDLE

    def handle_command(self, command: str) -> None:
        """Route a transcribed voice/text command to the right handler."""
        if not command:
            return

        if self.state == ConversationState.AWAITING_DRINK_DESCRIPTION:
            self.state = ConversationState.IDLE
            self._fulfill_drink_order(command)
            return

        if any(phrase in command for phrase in TRIGGER_PHRASES):
            self.voice.speak("Sure, what kind of drink would you like?")
            self.state = ConversationState.AWAITING_DRINK_DESCRIPTION
            return

        if any(phrase in command for phrase in EXIT_PHRASES):
            self.voice.speak("Goodbye!")
            self.on_exit()
            return

        response = self.recipes.chat(command)
        self.voice.speak(response)

    def _fulfill_drink_order(self, drink_request: str) -> None:
        """Generate a recipe for the request and dispense it, if possible."""
        available_ingredients = list(self.hardware.pin_map.keys())
        recipe = self.recipes.generate_recipe(drink_request, available_ingredients)

        if recipe.is_empty():
            self.voice.speak("Sorry, I couldn't find a recipe for that with what's available.")
            return

        recipe.title = self.recipes.generate_title(recipe)
        self.voice._on_status_update(f'--- Preparing: "{recipe.title}" ---')

        command = self.hardware.build_dispense_command(recipe)
        if not command:
            self.voice.speak("None of the required ingredients are available.")
            return

        try:
            self.voice._on_status_update("Sending recipe to dispenser...")
            confirmation = self.hardware.dispense(command)
            if confirmation == "DONE":
                self.voice.speak("Your drink is ready. Enjoy!")
            else:
                self.voice.speak(f"Unexpected response from the dispenser: '{confirmation}'.")
        except serial.SerialException:
            logger.exception("Serial communication with dispenser failed.")
            self.voice.speak("I couldn't connect to the dispenser. Please check the connection.")


# ------------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------------

class DispenserGUI:
    """Tkinter front-end providing text and microphone input alongside a
    scrolling conversation log."""

    def __init__(self, assistant: DispenserAssistant) -> None:
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("Voice-Controlled Beverage Dispenser")
        self.root.geometry("500x600")

        self._build_widgets()
        self.assistant.on_exit = self.root.quit
        self.assistant.voice._on_status_update = self.update_log

    def _build_widgets(self) -> None:
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.chat_log = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20)
        self.chat_log.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.text_input = tk.Entry(input_frame)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.text_input.bind("<Return>", self._on_text_submit)

        tk.Button(input_frame, text="Send", command=self._on_text_submit).pack(side=tk.LEFT, padx=5)
        self.mic_button = tk.Button(input_frame, text="🎤 Mic", command=self._on_mic_click)
        self.mic_button.pack(side=tk.LEFT, padx=5)

    def update_log(self, text: str) -> None:
        self.chat_log.insert(tk.END, text + "\n")
        self.chat_log.see(tk.END)

    def _on_text_submit(self, _event=None) -> None:
        command = self.text_input.get().strip().lower()
        if not command:
            return
        self.update_log(f"You: {command}")
        self.text_input.delete(0, tk.END)
        threading.Thread(target=self.assistant.handle_command, args=(command,), daemon=True).start()

    def _on_mic_click(self) -> None:
        threading.Thread(target=self._run_listening_flow, daemon=True).start()

    def _run_listening_flow(self) -> None:
        self.mic_button.config(state=tk.DISABLED)
        command = self.assistant.voice.listen()
        if command:
            self.assistant.handle_command(command)
        self.mic_button.config(state=tk.NORMAL)

    def run(self) -> None:
        threading.Thread(
            target=self.assistant.voice.speak,
            args=("The dispenser is ready for your commands.",),
            daemon=True,
        ).start()
        self.root.mainloop()


# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set. Please add it to your .env file.")
        return

    port = DispenserHardware.find_serial_port()
    if not port:
        logger.error("Could not find the dispenser board. Check the connection and try again.")
        return

    hardware = DispenserHardware(port=port)
    hardware.configure_ingredients(prompt_for_bar_setup())
    if not hardware.pin_map:
        logger.error("No ingredients configured. Exiting.")
        return

    voice = VoiceInterface()
    recipes = RecipeGenerator(api_key=api_key)
    assistant = DispenserAssistant(voice=voice, recipes=recipes, hardware=hardware)

    gui = DispenserGUI(assistant)
    gui.run()


if __name__ == "__main__":
    main()
