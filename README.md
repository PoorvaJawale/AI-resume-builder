# AI Resume Builder

## Overview

The AI Resume Builder is an innovative web application designed to assist users in creating professional resumes effortlessly. Developed for the Code-O-Fiesta coding competition, this project leverages Flash for the frontend interface and integrates Ollama's GPT-OSS-20B model to generate and enhance resume content intelligently.

## Features

- **Flash-Based Frontend**: A dynamic and interactive user interface built using Adobe Flash, providing a seamless experience for users to input their details and customize their resumes.
- **AI-Powered Content Generation**: Utilizes Ollama's GPT-OSS-20B model to generate personalized resume content, including professional summaries, skill descriptions, and experience highlights.
- **Customizable Templates**: Offers a variety of professionally designed resume templates that users can choose from and customize to suit their personal style and career objectives.
- **Real-Time Preview**: Allows users to see a real-time preview of their resume as they input their information, ensuring the final output meets their expectations.
- **Downloadable Formats**: Enables users to download their completed resumes in multiple formats, including PDF and DOCX, for easy submission to potential employers.

## Installation

To set up the AI Resume Builder locally, follow these steps:

### Prerequisites

- Adobe Flash Player installed on your system.
- Python 3.x installed.
- Required Python libraries listed in `requirements.txt`.

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/PoorvaJawale/AI-resume-builder.git
   cd AI-resume-builder
   
2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt

4. Start the Flask application:

   ```bash
   python app.py

5. Open your browser and navigate to http://127.0.0.1:5000 to access the AI Resume Builder.

### Usage
1. Launch the application and select a resume template.
2. Input your personal details, including name, contact information, education, skills, and work experience.
3. Use the AI-powered suggestions to enhance your resume content.
4. Preview your resume in real-time.
5. Once satisfied, download your resume in the desired format.

### AI Integration
The AI functionalities are powered by Ollama's GPT-OSS-20B model. This model assists in generating and refining resume content, ensuring that each resume is tailored to highlight the user's strengths and experiences effectively.

### Contributing
Contributions to the AI Resume Builder project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Create a new Pull Request.

### License
This project is licensed under the MIT License 
