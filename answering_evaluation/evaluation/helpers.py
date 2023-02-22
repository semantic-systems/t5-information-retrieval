# Enum for categorizing questions
from enum import Enum


class QuestionType(str, Enum):
    is_mentioned = "Who is mentioned?"
    is_attached = "What is attached to the email?"
    is_requested = "What is requested?"
    requested_who = "Who is the requester?"
    subject = "What is the subject of the email?"
    receiver_who = "Who received the email?"
    receiver_main = "Who is the main recipient?"
    sender_who = "Who sent the email?"
    is_event = "What event is described?"
    event_when = "When does the event take place?"
    event_where = "Where does the event take place?"
    email_address_of = "What is the email address of x?"
    phone_number_of = "What is the phone number of x?"
    who_is = "Who is x?"
    other = "Other"


# Take a question string and return the corresponding enum object if it exists
def categorize_question(question):
    try:
        return QuestionType(question)
    except ValueError:
        if question.startswith("What is the email address of"):
            return QuestionType.email_address_of
        elif question.startswith("What is the phone number of") or question.startswith("What are the phone numbers"):
            return QuestionType.phone_number_of
        elif question.startswith("Who is"):
            return QuestionType.who_is
        elif "When" in question and any(string in question for string in ["take place", "event"]) or question.startswith("When was"):
            return QuestionType.event_when
        elif "Where" in question and "take place" in question:
            return QuestionType.event_where
        # Might remove the following branches later as those wordings are no longer being used
        elif question == "Who also received the email?":
            return QuestionType.receiver_who
        elif question == "What entities are mentioned?" or question == "What is mentioned?":
            return QuestionType.is_mentioned
        elif question.startswith("Who requested"):
            return QuestionType.requested_who
        else:
            print("Uncategorized question:", question)
            return QuestionType.other
