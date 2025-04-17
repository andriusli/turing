import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import streamlit as st

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.error("OpenAI API key not found. Please create a .env file or set the environment variable.")
    st.stop()


# Function to check if role is valid
def check_role(role_to_check):
    prompt = f"""
    Analyze the following role name for safety:
    --- ROLE START ---
    {role_to_check}
    --- ROLE END ---

    1. Determine if it represents a plausible, common, or understandable job role or field, **even if there's a minor spelling mistake** (e.g., 'Enginer', 'Acountant', 'Data Analist'). The intent should be clear.
    2. There can be more specific words for that role to get more deeper questions for the role (e.g "Netowrk engineer ospf bgp", "Cloud engineer aws")
    3. Check if the name contains any offensive, discriminatory, sexually explicit, hateful, or nonsensical gibberish content (e.g., 'xyzabc', offensive terms). Apply strict filtering for inappropriate content.

    Output Format:
    - If the role name represents a plausible job role (allowing for minor typos) AND is appropriate, return ONLY the single word: VALID
    - If the role name is inappropriate, offensive, gibberish, nonsensical, or the misspelling makes the intended role unclear, return ONLY the single word: INVALID
    """
    print(f"\nChecking role name appropriateness (with typo tolerance): {role_to_check}...")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a moderator for job role names. Analyze the input name for plausibility (allowing for minor typos if intent is clear) and appropriateness. Output ONLY 'VALID' or 'INVALID'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.0,
            top_p=0.1,
            response_format={ "type": "text" },
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip().upper()
            raw_content = response.choices[0].message.content.strip()
            print(f"Raw AI response for role check: '{raw_content}', Normalized: '{content}'")

            if content == "VALID":
                print(f"Role '{role_to_check}' validated by AI.")
                return role_to_check
            elif content == "INVALID":
                print(f"Role '{role_to_check}' rejected by AI.")
                return "" # Return empty string for rejected role
            else:
                # returned something other than VALID or INVALID
                print(f"Warning: AI returned unexpected content '{raw_content}'. Treating as error.")
                st.error(f"Error checking role: AI response was unclear ('{raw_content}'). Please try again.")
                return None

        else:
            print("Error: No valid response choices received from OpenAI for role check.")
            st.error("Error checking role name: Invalid response structure from AI.")
            return None

    except Exception as e:
        print(f"\nAn unexpected error occurred during role name validation: {e}")
        st.error(f"An unexpected error occurred during role name validation: {e}")
        return None


# Function to generate questions using OpenAI
def generate_questions_openai(num_questions, complexity_of_question, role="General"):
    print(f"\nGenerating {num_questions} questions for role: {role}...")

    prompt = f"""
    Generate exactly {num_questions} **{complexity_of_question}** interview questions specifically tailored for a **{role}** position. 
    The questions should assess relevant technical skills (if applicable), problem-solving abilities, experience, and professional approach related to the **{role}** field.
    Ensure the questions cover a diverse range of scenarios and challenges relevant to the role and make sure you base them heavily on the skills, tools, and responsibilities

    Specific Role Instructions:
    - If the role is anything but "General":
        - Approximately 20% of questions should be general behavioral/professional questions.
        - General questions should preferably appear first.
        - The remaining 80% must be specifically tailored to the skills and responsibilities of a **{role}** position.
        - Generate questions with varied formats (e.g., situational, technical deep-dive, design
    - If the role is "General":
        - Generate broad professional questions suitable for a wide range of roles, focusing on experience, problem-solving, teamwork, and career goals, while respecting the constraints below.

    IMPORTANT CONSTRAINTS (Apply to ALL roles):
    1. DO NOT ask the following specific, generic questions:
       - "What are your strengths?"
       - "What are your weaknesses?"
       - "Where do you see yourself in 5 years?"
       - "Why should we hire you?"
    2. Absolutely DO NOT ask any questions related to:
       - Religion or religious beliefs/practices
       - Sexual orientation or gender identity
       - Political affiliations or views
       - Health conditions, disabilities, or medical history
       - Personal family matters (marital status, children, pregnancy plans etc.)
       - Age (unless directly job-related and legally permissible, which is rare)
       - Race or ethnicity
       - National origin or citizenship status (beyond legal work authorization)
    3. Make sure there are {num_questions} **{complexity_of_question}** interview questions specifically tailored for a **{role}** position.

    Output Format:
    Return ONLY a valid **JSON object** containing a single key "questions" whose value is a list of strings (the interview questions).
    Example for num_questions=2, role="App Developer":
    {{
      "questions": ["Describe a challenging technical problem you solved recently and your approach.", "How do you ensure the quality and maintainability of your code?"]
    }}
    Ensure the entire output is a single, valid JSON object starting with '{{' and ending with '}}'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"You are an expert assistant that generates professional interview questions tailored for a specific job role ({role}). You output *only* a valid JSON object containing a list of questions under the 'questions' key, strictly following all constraints provided."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=150 * num_questions,
            temperature=0.9,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.9,
        )

        content = response.choices[0].message.content
        print(f"\nRaw JSON response received from OpenAI (Questions):\n{content}\n")

        try:
            data = json.loads(content)
            # Validate the structure
            if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                # Extract the list of questions
                parsed_questions = data["questions"]
                # Optional: Further validation if needed (e.g., check if all items are strings)
                if all(isinstance(q, str) for q in parsed_questions):
                    print(f"Successfully parsed {len(parsed_questions)} questions from JSON.")
                    # Ensure there are the same number of questions as requested
                    return parsed_questions[:num_questions]
                else:
                    print("Error: Not all items in the 'questions' list are strings.")
                    st.error("Error: AI returned invalid data format inside the JSON 'questions' list.")
                    return []
            else:
                print("Error: JSON response missing 'questions' key or value is not a list.")
                st.error("Error: AI response did not contain the expected 'questions' list in the JSON object.")
                return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Faulty JSON string: {content}")
            st.error(f"Error: Could not parse the AI's response as valid JSON.")
            return []

    except Exception as e:
        # Check for other API errors or issues
        print(f"\nAn unexpected error occurred during question generation: {e}")


# Function to check answers fo safety
def check_answer_safety(answer_text):
    # Handle empty input fist
    if not answer_text or not answer_text.strip():
        print("Answer is empty, considered SAFE by default.")
        return True

    # Focused prompt for safety analysis of a single answer
    prompt = f"""
    Analyze the following text for safety:
    --- TEXT START ---
    {answer_text}
    --- TEXT END ---

    Instructions:
    Check STRICTLY if the text contains ANY offensive, discriminatory, sexually explicit, hateful, harmful, or nonsensical gibberish content.
    Examples include slurs, threats, graphic descriptions, hate speech, harassment, etc. Do NOT tolerate ANY inappropriate content.
    Focus only on the presence of unsafe content.

    Output Format:
    - If the text is SAFE (contains NO inappropriate content), return ONLY the single word: SAFE
    - If the text is UNSAFE (contains ANY inappropriate content), return ONLY the single word: UNSAFE
    """

    try:
        # Ensure client is available - might need to pass it or ensure it's global
        if 'client' not in globals() or not isinstance(client, OpenAI):
             st.error("OpenAI client not initialized for safety check.")
             return None

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a strict content moderator. Analyze the provided text STRICTLY for any offensive, discriminatory, sexually explicit, hateful, harmful, or nonsensical gibberish content. Output ONLY 'SAFE' or 'UNSAFE'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.0,
            top_p=0.1,
            response_format={ "type": "text" },
            frequency_penalty=0,
            presence_penalty=0,
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip().upper()
            raw_content = response.choices[0].message.content.strip()
            print(f"Raw AI safety response: '{raw_content}', Normalized: '{content}'")

            if content == "SAFE":
                print("Answer seemed SAFE by AI.")
                return True
            elif content == "UNSAFE":
                print("Answer deemed UNSAFE by AI.")
                return False
            else:
                # If AI returned something unexpected
                print(f"Warning: AI returned unexpected safety content '{raw_content}'. Treating as unclear.")
                st.warning(f"Safety check unclear: AI response was '{raw_content}'. Review manually if possible.")
                return False # Treat unclear answers as unsafe

        else:
            print("Error: No valid response choices received from OpenAI for answer safety check.")
            st.error("Error checking answer safety: Invalid response structure from AI.")
            return None


    except Exception as e:
        print(f"\nAn unexpected error occurred during answer safety check: {e}")
        st.error(f"An unexpected error occurred during answer safety check: {e}")
        return None


# Function to Evaluate Answers using AI
def evaluate_answers_openai(questions, answers, num_questions, role="General"):
    print(f"\nAttempting to evaluate answers for role: {role}...")
    if not questions or not answers or len(questions) != len(answers):
         print("Evaluation skipped: Invalid questions or answers provided.")
         return None

    # Combine Q&A pairs into a transcript format
    interview_transcript = ""
    valid_answers_provided = False
    for i, (q, a) in enumerate(zip(questions, answers)):
        answer_text = a.strip() if a else "" # Get answer text, default to empty string if None
        interview_transcript += f"Question {i+1}: {q}\n"
        if answer_text:
            interview_transcript += f"Answer {i+1}: {answer_text}\n\n"
            valid_answers_provided = True
        else:
             # Mark unanswered questions
             interview_transcript += f"Answer {i+1}: --- NOT ANSWERED ---\n\n"

    # If no answers were typed don't call API
    if not valid_answers_provided:
        print("Evaluation skipped: No actual answers were provided by the user.")
        st.warning("Cannot evaluate as no answers were provided.")
        # Return a structured response indicating no answers
        return {
          "evaluations": [
            {"question_index": i, "grade": 1, "justification": "Not answered"} for i in range(len(questions))
          ],
          "overall_grade": 1,
          "overall_justification": "No answers were provided for evaluation."
        }


    # The prompt for the evaluation
    evaluation_prompt = f"""
    Act as an expert hiring manager and strict interviewer evaluating a candidate's performance for a **{role}** position based on the provided interview transcript.

    Transcript:
    --- Transcript Start ---
    {interview_transcript}
    --- Transcript End ---

    Your Task:
    1.  **Evaluate Each Answer:** For every question in the transcript:
        * Provide a numerical **grade** from 1 (Poor) to 10 (Excellent).
        * Provide a concise **justification** (1-2 sentences) for the grade. Base the evaluation on:
            * **Relevance:** Does the answer directly address the question?
            * **Clarity:** Is the answer clear, well-structured, and easy to understand?
            * **Depth & Detail:** Does the answer provide sufficient detail and examples?
            * **Role Appropriateness:** Is the content and level of detail appropriate for a candidate applying for a **{role}** position?
    2.  **Handle Unanswered Questions:** If a question is marked as "--- NOT ANSWERED ---", assign a grade of **1** and use the justification "**Not answered**".
    3.  **Content Safety:** If an answer contains offensive, discriminatory, inappropriate content, or is completely irrelevant gibberish, assign a grade of **1** and use the justification "**Inappropriate or irrelevant content**". Do not evaluate the substance otherwise.
    4.  **Overall Assessment:** After evaluating all individual answers:
        * Provide an **overall_grade** (1-10) reflecting the candidate's performance across the entire interview.
        * Provide an **overall_justification** (2-4 sentences) summarizing strengths and weaknesses, **specifically in the context of the {role} role requirements**. Mention potential suitability or areas needing significant improvement for this type of position.

    Output Format:
    Return **ONLY** a valid **JSON object** adhering strictly to the following structure. Do not include any text before or after the JSON object.
    {{
      "evaluations": [
        {{
          "question_index": integer, // Index of the question (0-based)
          "grade": integer,          // Grade for this answer (1-10)
          "justification": "string"  // Justification (or "Not answered", "Inappropriate or irrelevant content")
        }}
        // ... one object for EACH question asked, matching the order in the transcript
      ],
      "overall_grade": integer,      // Overall interview grade (1-10)
      "overall_justification": "string" // Overall feedback summary tailored to the role
    }}

    Ensure the 'evaluations' list contains exactly one entry for each question asked, maintaining the original order.
    """

    try:
        # Make the API call for evaluation, requesting JSON output
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an AI evaluation assistant. You analyze interview transcripts for a '{role}' role and provide structured feedback strictly in the specified JSON format."},
                {"role": "user", "content": evaluation_prompt}
            ],
            response_format={ "type": "json_object" },
            top_p=1,
            max_tokens=200 * num_questions,
            temperature=0.6,
            frequency_penalty=0.2,
            presence_penalty=0.2,
        )
        evaluation_content = response.choices[0].message.content
        print(f"\nRaw response received from OpenAI (Evaluation):\n{evaluation_content}\n")

        evaluation_results = json.loads(evaluation_content)

        # Validation of the received JSON
        if not isinstance(evaluation_results, dict):
            raise ValueError("Evaluation response is not a JSON object.")
        if "evaluations" not in evaluation_results or not isinstance(evaluation_results["evaluations"], list):
            raise ValueError("JSON response missing 'evaluations' list.")
        if "overall_grade" not in evaluation_results or not isinstance(evaluation_results["overall_grade"], int):
             raise ValueError("JSON response missing or invalid 'overall_grade'.")
        if "overall_justification" not in evaluation_results or not isinstance(evaluation_results["overall_justification"], str):
             raise ValueError("JSON response missing or invalid 'overall_justification'.")
        if len(evaluation_results["evaluations"]) != len(questions):
             raise ValueError(f"JSON response 'evaluations' list length ({len(evaluation_results['evaluations'])}) does not match number of questions ({len(questions)}).")
        # Checking structure for each item in 'evaluations'
        for i, item in enumerate(evaluation_results["evaluations"]):
             if not isinstance(item, dict) or \
                "question_index" not in item or item["question_index"] != i or \
                "grade" not in item or not isinstance(item["grade"], int) or \
                "justification" not in item or not isinstance(item["justification"], str):
                 raise ValueError(f"Invalid structure for evaluation item at index {i}.")

        print("Successfully parsed and validated evaluation results.")
        return evaluation_results

    except json.JSONDecodeError as json_err:
         st.error(f"Error: Could not parse the AI's evaluation response (invalid JSON). {json_err}")
         print(f"Error parsing evaluation JSON: {json_err}")
         print(f"Faulty JSON string attempt: {evaluation_content}")
         return None
    except ValueError as val_err:
         st.error(f"Error: The AI's evaluation response had an invalid structure. {val_err}")
         print(f"Error validating evaluation JSON structure: {val_err}")
         return None
    except Exception as e:
        # Other potential API errors or issues
        st.error(f"An unexpected error occurred during evaluation: {e}")
        print(f"Error in evaluate_answers_openai: {e}")
        return None


# --- Streamlit App Layout and Logic ---
APP_TITLE = "AI Interview Assistant"

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.write("Practice interviewing with AI questions tailored to a role and get AI feedback.")

# --- Initialize Session State ---
st.session_state.setdefault('interview_phase', 'setup')
st.session_state.setdefault('questions', [])
st.session_state.setdefault('answers', [])
st.session_state.setdefault('current_question_index', 0)
st.session_state.setdefault('evaluation_results', None)
st.session_state.setdefault('num_questions_selected', 5)
st.session_state.setdefault('questions_complexity', 'Medium')
st.session_state.setdefault('selected_option', 'App Developer')
st.session_state.setdefault('custom_role_input', '')
st.session_state.setdefault('effective_role', 'App Developer')


# === Phase 1: Setup ===
if st.session_state.interview_phase == 'setup':
    st.header("Configure Your Mock Interview")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Number of questions dropdown
        question_options = list(range(3, 11))
        st.session_state.num_questions_selected = st.selectbox(
            "Number of Questions:",
            options=question_options,
            key='num_q_select',
            index=question_options.index(st.session_state.num_questions_selected)
        )

    with col2:
        # Role selection dropdown and custom input
        role_options = ["App Developer", "Data Analyst", "Big Data Engineer", "General", "Other..."]
        try:
            # Determine initial index based on last known effective role
            if st.session_state.effective_role not in role_options:
                 current_selection_index = role_options.index("Other...")
            else:
                 current_selection_index = role_options.index(st.session_state.effective_role)
        except ValueError:
            current_selection_index = role_options.index("App Developer") # Default fallback

        st.session_state.selected_option = st.selectbox(
            "Interview Type:",
            options=role_options,
            key='role_select',
            index=current_selection_index
        )

    with col3:
        # Complexity of the interview
        question_options = ["Easy", "Medium", "Hard"]
        st.session_state.questions_complexity = st.selectbox(
            "Complexity of Questions:",
            options=question_options,
            key='q_complexity',
            index=question_options.index(st.session_state.questions_complexity)
        )


        # Custom role input field logic
        custom_role = ""
        if st.session_state.selected_option == "Other...":
            st.session_state.custom_role_input = st.text_input(
                "Enter Custom Role:",
                key="custom_role_text",
                value=st.session_state.custom_role_input,
                max_chars=50,
            ).strip()
            custom_role = st.session_state.custom_role_input
        else:
             st.session_state.custom_role_input = "" # Clear if not "Other..."

        # Determine the potentially effective role based on selection
        if st.session_state.selected_option == "Other...":
            st.session_state.effective_role = custom_role if custom_role else "Other..."
        else:
            st.session_state.effective_role = st.session_state.selected_option

    # --- Role Display and Validation Logic ---
    display_role = st.session_state.effective_role
    effective_role_for_start = "" # This will hold the validated role name, or remain empty

    if display_role == "Other...":
         st.info("Please specify the custom role above.")
         # effective_role_for_start remains "" -> button disabled
    elif not display_role: # Handles case where 'Other...' selected but input cleared/empty
        st.info("Please specify the custom role above.")
        # effective_role_for_start remains "" -> button disabled
    else:
        # A role name is present, proceed to validate it using the revised check_role
        placeholder = st.empty() # Use a placeholder for messages during check
        placeholder.write(f"Checking role: **{display_role}**...")

        # Call validation function within a spinner
        with st.spinner(f"Validating role name '{display_role}'..."):
             # Use the revised check_role function
             checked_role_name = check_role(display_role) # Returns string (original name), "", or None

        # --- Handle Validation Result ---
        placeholder.empty() # Clear the "Checking..." message

        if checked_role_name is not None: # Check if validation call itself succeeded (didn't return None)
             if checked_role_name != "": # Check if role was deemed valid (AI returned VALID -> function returned original name)
                 effective_role_for_start = checked_role_name # Role is valid, use the validated name
                 st.success(f"Role '{checked_role_name}' is valid.") # Confirmation message
                 st.write(f"Ready to start a **{checked_role_name}** interview with **{st.session_state.num_questions_selected}** questions.")
                 # Optional: Update session state if validation somehow changed the name (e.g., case normalization - though we return original now)
                 # st.session_state.effective_role = checked_role_name
             else: # Role was deemed invalid by AI (AI returned INVALID -> function returned "")
                 st.error(f"The role name '{display_role}' was deemed inappropriate or not a valid job role. Please enter a different one.")
                 # effective_role_for_start remains "" (button disabled)

        else: # Validation call failed or returned unclear response (check_role returned None)
             # Error message is displayed by check_role or the calling logic here
             st.warning("Could not confirm role validity. Please try again or choose a different role.")
             # effective_role_for_start remains "" (button disabled)

    # Button to start the interview - enabled only if effective_role_for_start is not empty
    start_button_label = f"Start {effective_role_for_start} Interview" if effective_role_for_start else "Start Interview"
    if st.button(start_button_label, disabled=(not effective_role_for_start)):
        # The role is already validated if effective_role_for_start is not empty
        final_role = effective_role_for_start
        num_to_generate = st.session_state.num_questions_selected
        complexity_of_question = st.session_state.questions_complexity

        with st.spinner(f"AI is preparing {num_to_generate} {complexity_of_question} questions for a {final_role}..."):
            st.session_state.questions = generate_questions_openai(num_to_generate, complexity_of_question, final_role)

        if st.session_state.questions:
            # Initialize answers list, reset index, clear old results
            st.session_state.answers = [""] * len(st.session_state.questions)
            st.session_state.current_question_index = 0
            st.session_state.evaluation_results = None
            # Change phase and rerun
            st.session_state.interview_phase = 'interviewing'
            st.rerun()
        else:
            # Error message is usually handled within generate_questions_openai
            st.error("Failed to generate interview questions. Please check the role or try again.")


# === Phase 2: Interviewing ===
elif st.session_state.interview_phase == 'interviewing':
    # Use the 'effective_role' stored from the setup phase
    current_role = st.session_state.effective_role
    st.header(f"{current_role} Interview in Progress...")

    total_questions = len(st.session_state.questions)
    q_index = st.session_state.current_question_index

    # Defensive check in case questions list is somehow empty
    if total_questions == 0 or q_index >= total_questions:
        st.error("Error: No questions found to display. Returning to setup.")
        st.session_state.interview_phase = 'setup'
        st.rerun()

    current_question = st.session_state.questions[q_index]

    # Display progress bar and question number/text
    st.progress((q_index) / total_questions) # Progress is 0-based index / total
    st.write(f"Question {q_index + 1} of {total_questions}")
    st.subheader(f"{q_index+1}. {current_question}")

    # Text Area with Character Limit and Counter ---
    answer = st.text_area(
        "Your Answer:",
        key=f"answer_{q_index}",
        value=st.session_state.answers[q_index], # Show previously entered answer
        height=250, # Adjusted height slightly
        max_chars=1000 # Set maximum character limit
    )
    # Store the potentially truncated answer back into session state
    # Streamlit handles the truncation automatically based on max_chars
    st.session_state.answers[q_index] = answer

    # Display interactive character count
    char_count = len(answer)
    # Use markdown for potentially better styling control if needed
    st.caption(f"Characters: {char_count} / 1000")
    # --- END OF MODIFICATION ---

    # Navigation buttons
    col1, col2, col3, col4 = st.columns([1, 1.5, 1, 1]) # Adjust ratios for button text length

    with col1:
        # "Previous" button - only show if not the first question
        if q_index > 0:
             if st.button("Previous"):
                 st.session_state.current_question_index -= 1
                 st.rerun()
        else:
            st.write("") # Placeholder to maintain layout

    with col2:
         # "Next" or "Finish" button
         is_last_question = (q_index == total_questions - 1)
         next_button_label = "Finish Interview & Evaluate" if is_last_question else "Submit & Next"
         next_button_type = "primary" if is_last_question else "secondary"

         if st.button(next_button_label, type=next_button_type):
             if is_last_question:
                  # Move to evaluation phase
                  st.session_state.interview_phase = 'finished'
             else:
                  # Move to the next question
                  st.session_state.current_question_index += 1
             st.rerun()

    with col3:
        # Button to end the interview early
        if st.button("End Early"):
             st.session_state.interview_phase = 'finished'
             st.rerun()

# === Phase 3: Finished & Evaluation ===
elif st.session_state.interview_phase == 'finished':
    # Use the 'effective_role' stored from the setup phase
    current_role = st.session_state.effective_role
    st.header(f"Interview Finished ({current_role})")

    all_answers_safe = True
    unsafe_indices = []
    for i, answer in enumerate(st.session_state.answers):
        if answer and answer.strip(): # Only check non-empty answers
            is_safe = check_answer_safety(answer)
            if is_safe is False: # Explicitly check for False (unsafe or unclear)
                all_answers_safe = False
                unsafe_indices.append(i)
                st.warning(f"Warning: Answer to question {i+1} flagged as potentially unsafe.")
                # Optionally: Replace the answer with a placeholder in the list sent for evaluation
                st.session_state.answers[i] = "[Content Flagged as Unsafe]"
            elif is_safe is None:
                # Handle API error case if needed, maybe skip evaluation?
                st.error(f"Could not verify safety for answer {i+1} due to an error.")
                all_answers_safe = False # Treat error as potentially unsafe

    if not all_answers_safe:
        st.error("Some answers were flagged as potentially unsafe and may not be evaluated properly.")

    # --- Trigger or Display Evaluation ---
    st.subheader("AI Evaluation:")

    # Check if evaluation results already exist in session state
    if st.session_state.evaluation_results is None:
        # If not evaluated yet, call the evaluation function
        # Check if there are actually answers to evaluate
        if any(a and a.strip() for a in st.session_state.answers): # Check if any answer is non-empty
             with st.spinner(f"AI is evaluating your answers for the {current_role} role... Please wait."):
                 # Pass the actual role used for the interview
                 st.session_state.evaluation_results = evaluate_answers_openai(
                     st.session_state.questions,
                     st.session_state.answers,
                     st.session_state.num_questions_selected,
                     current_role, # Pass the role context
                 )
                 # Check if evaluation actually returned results before rerunning
                 if st.session_state.evaluation_results:
                     st.rerun() # Rerun to display results now that they exist
                 else:
                     # If evaluation failed, error is shown by evaluate_answers_openai
                     st.error("Failed to get evaluation results from the AI.")
        else:
            st.warning("No answers were provided during the interview, so no evaluation can be performed.")
            # Create a default "no results" structure to prevent errors below
            st.session_state.evaluation_results = {
                "evaluations": [{"question_index": i, "grade": 1, "justification": "Not answered"} for i in range(len(st.session_state.questions))],
                "overall_grade": 1,
                "overall_justification": "No answers were provided for evaluation."
            }

    # --- Display Evaluation Results (if available) ---
    if st.session_state.evaluation_results:
        eval_data = st.session_state.evaluation_results

        # Display Overall Feedback first
        st.markdown("---")
        st.markdown(f"#### Overall Performance (for {current_role})")

        overall_grade = eval_data.get('overall_grade', 'N/A')
        col_grade, col_feedback = st.columns([1, 3])
        with col_grade:
             # Ensure grade is displayed correctly even if N/A
             grade_display = f"{overall_grade} / 10" if isinstance(overall_grade, int) else "N/A"
             st.metric(label="Overall Grade", value=grade_display)
        with col_feedback:
             st.info(f"**Overall Feedback:** {eval_data.get('overall_justification', 'No overall feedback provided.')}")
        st.markdown("---")

        # Display Detailed Feedback per Question
        st.subheader("Detailed Feedback per Question:")
        evaluations_list = eval_data.get('evaluations', [])

        if len(evaluations_list) == len(st.session_state.questions):
             for i, question in enumerate(st.session_state.questions):
                 st.markdown(f"**Q{i+1}: {question}**")
                 answer = st.session_state.answers[i] if i < len(st.session_state.answers) else ""
                 # Use an expander for cleaner look, default to collapsed unless not answered
                 is_answered = answer and answer.strip()
                 with st.expander(f"Your Answer & AI Feedback (Q{i+1})", expanded=(not is_answered)):
                    st.markdown(f"> _Your Answer: {answer if is_answered else '(Not answered)'}_")

                    # Ensure evaluation exists for index i
                    if i < len(evaluations_list):
                        evaluation = evaluations_list[i]
                        grade = evaluation.get('grade')
                        justification = evaluation.get('justification')

                        if grade is not None and justification:
                            # Use columns for grade and justification
                            fb_col1, fb_col2 = st.columns([1, 4])
                            with fb_col1:
                                st.metric(label="Grade", value=f"{grade}/10")
                            with fb_col2:
                                # Apply different styling based on grade or justification
                                if grade <= 1 and justification in ["Not answered", "Inappropriate or irrelevant content"]:
                                    st.warning(f"**AI Feedback:** {justification}")
                                elif grade < 5:
                                    st.warning(f"**AI Feedback:** {justification}")
                                else:
                                    st.success(f"**AI Feedback:** {justification}")
                        else:
                            st.warning("Evaluation data (grade/justification) missing for this question.")
                    else:
                        st.warning("Evaluation data missing for this question index.")
                 st.divider() # Add a visual separator between questions
        else:
            st.error("Mismatch between the number of questions and the number of evaluations received. Cannot display detailed feedback reliably.")
            print(f"Data mismatch: {len(st.session_state.questions)} questions, {len(evaluations_list)} evaluations.")

    else:
        # This case might be hit if evaluation failed critically after the spinner
        st.error("Could not retrieve or display AI evaluation for your answers.")

    # --- Option to Start Over ---
    st.markdown("---")
    if st.button("Start New Interview"):
        # Reset relevant state variables to defaults for a fresh start
        st.session_state.interview_phase = 'setup'
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.current_question_index = 0
        st.session_state.evaluation_results = None
        st.session_state.selected_option = 'App Developer'
        st.session_state.custom_role_input = ''
        st.session_state.effective_role = 'App Developer'
        st.session_state.num_questions_selected = 3
        st.rerun()

# --- Simple Sidebar Info ---
# st.sidebar.image("YOUR_LOGO_URL_HERE", width=200) # Optional logo
st.sidebar.info("Configure your mock interview settings in the main panel, then click 'Start Interview'.")
if not api_key:
    st.sidebar.error("OpenAI API Key Missing!")
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by OpenAI & Streamlit")
