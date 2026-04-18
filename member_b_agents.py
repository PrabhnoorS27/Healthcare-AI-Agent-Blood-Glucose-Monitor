from crewai import Agent, Task
from member_a_data import llm, get_blood_glucose

PATIENT_AGE = None

data_agent = Agent(
    role="Health Data Fetcher",
    goal="Get the blood glucose and health data of a patient from our dataset",
    backstory=(
        "you are a helpful medical assistant who gets patient health "
        "records from a database. you have access to real clinical data "
        "with 16969 patient records. you get the data accurately and "
        "pass it to the doctor for analysis."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    memory=True,
    max_iter=1
)

analysis_agent = Agent(
    role="Health Advisor",
    goal="look at the patient health data and tell what it means and give advice",
    backstory=(
        "you are a virtual AI doctor who specializes in diabetes and "
        "blood glucose monitoring. you look at the patient readings and "
        "explain in simple english what everything means. "
        "you give practical tips the patient can follow at home."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    memory=True,
    max_iter=1
)

fetch_task = Task(
    description=(
        f"get a real patient health record from our kaggle dataset. "
        f"{'find a patient who is ' + str(PATIENT_AGE) + ' years old.' if PATIENT_AGE else 'pick any random patient.'} "
        f"return all their health information."
    ),
    agent=data_agent,
    execute_fn=lambda: get_blood_glucose(PATIENT_AGE),
    expected_output=(
        "patient health summary with blood glucose, blood pressure, "
        "heart rate, temperature, oxygen level, and diabetic status"
    )
)

analyze_task = Task(
    description=(
        "look at the patient health data you received and do these things:\n"
        "1. check blood glucose level:\n"
        "   - below 70 = too low\n"
        "   - 70 to 99 = normal\n"
        "   - 100 to 125 = pre-diabetic\n"
        "   - 126 or above = diabetic\n"
        "2. check blood pressure (normal is 120/80)\n"
        "3. check heart rate (normal is 60 to 100)\n"
        "4. check oxygen level (normal is above 95%)\n"
        "5. note sweating and shivering\n"
        "6. give 5 simple health tips\n"
        "7. say if patient needs to see a doctor urgently"
    ),
    agent=analysis_agent,
    expected_output=(
        "a full health report with glucose category, "
        "blood pressure status, heart rate status, oxygen level, "
        "symptoms, 5 health tips, and doctor advice"
    )
)