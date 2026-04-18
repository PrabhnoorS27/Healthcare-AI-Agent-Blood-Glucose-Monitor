

from crewai import Crew
from member_b_agents import (
    data_agent,
    analysis_agent,
    fetch_task,
    analyze_task,
    PATIENT_AGE
)


crew = Crew(
    agents=[data_agent, analysis_agent],
    tasks=[fetch_task, analyze_task],
    verbose=True
)

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("  HEALTHCARE AI PROJECT")
    print("  Blood Glucose Monitor using CrewAI")
    print("  Dataset: Kaggle (16969 real patients)")
    print("  Model: Groq Llama3 (free!)")
    print("=" * 50)
    print()

    if PATIENT_AGE:
        print(f"analyzing patient aged {PATIENT_AGE} years...")
    else:
        print("picking a random patient from dataset...")
    print()

    query = {
        "query": (
            "please get a patient record from the dataset "
            "and give me a complete health report"
        )
    }

    response = crew.kickoff(query)

    print()
    print("=" * 50)
    print("  AI HEALTH REPORT")
    print("=" * 50)
    print(response)
    print()
    print("done!! hope the AI advice was helpful :)")