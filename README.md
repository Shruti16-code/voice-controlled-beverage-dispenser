# Voice-Controlled Automated Beverage Dispenser

An automated beverage dispensing system controlled entirely through speech, enabling hands-free drink selection. Built for accessibility — improving usability for differently-abled users and reducing physical contact for better hygiene.

## Features

- **Voice-controlled ordering** — describe a drink out loud, no touchscreen or buttons needed
- **Real-time audio feedback** — guides the user through selection and dispensing via spoken responses
- **AI-generated recipes** — an LLM builds a recipe from the request, constrained to whatever ingredients are physically loaded
- **Hardware dispensing** — recipe is sent to an STM32 microcontroller, which drives the pumps

## Tech Stack

- **Hardware**: STM32 microcontroller (serial communication)
- **Language**: Python
- **Speech-to-text**: Google Speech Recognition (`SpeechRecognition`)
- **Text-to-speech**: `gTTS`
- **Recipe generation**: LLM via OpenRouter API
- **GUI**: Tkinter

## Setup

1. Clone the repository
   \`\`\`bash
   git clone https://github.com/Shruti16-code/voice-controlled-beverage-dispenser.git
   cd voice-controlled-beverage-dispenser
   \`\`\`

2. Install dependencies
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Set up your environment variables
   - Copy `.env.example` to `.env`
   - Add your OpenRouter API key inside `.env`

4. Connect the STM32 dispenser board via USB

5. Run the application
   \`\`\`bash
   python src/beverage_dispenser.py
   \`\`\`

6. On first run, you'll be asked which ingredient is loaded on each pump pin (4–7)

## Usage

- Click the 🎤 button or type a command
- Say something like *"make me a drink"* — the assistant will ask what kind
- Describe what you want (e.g. *"something fruity and sweet"*)
- The assistant generates a recipe, names it, and dispenses it automatically

## Project Motivation

This project was built to make beverage service more accessible and hygienic — removing the need for physical interaction with buttons or touchscreens, and giving differently-abled users an easier way to order and receive a drink independently.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
