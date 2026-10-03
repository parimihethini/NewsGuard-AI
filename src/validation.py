import re
import string

GREETINGS_AND_CASUAL = {
    "hello", "hi", "hey", "how are you", "good morning", "good evening",
    "good afternoon", "good night", "thanks", "thank you", "bye", "goodbye",
    "test", "testing", "who are you", "what is your name", "help me"
}

def validate_news_input(text: str) -> tuple[bool, str]:
    """
    Conservative input validation layer for NewsGuard AI.
    Ensures input is a legitimate news headline, claim, or article before
    passing it to deep learning model or explanation engines.

    Returns:
        (is_valid, message)
    """
    if text is None:
        return False, "Input cannot be empty. Please enter a news headline, claim, or article for analysis."

    cleaned_raw = text.strip()
    if not cleaned_raw:
        return False, "Input cannot be empty. Please enter a news headline, claim, or article for analysis."

    # Minimum character length
    if len(cleaned_raw) < 12:
        return False, "Input is too short (minimum 12 characters). Please enter a news headline, claim, or article for analysis."

    # Check for pure or predominant math / equation expressions (e.g. '2+2=3', '10 * 5 = 50')
    if re.match(r"^[\d\s\+\-\*\/\^=%\(\)\.\<\>!]+$", cleaned_raw):
        return False, "Mathematical expressions and calculations are unsupported. Please enter a news headline, claim, or article for analysis."

    # Check for digit-heavy content
    alpha_chars = sum(c.isalpha() for c in cleaned_raw)
    total_non_space = sum(not c.isspace() for c in cleaned_raw)
    if total_non_space > 0 and (alpha_chars / total_non_space) < 0.40:
        return False, "Input contains predominantly symbols or numbers. Please enter a news headline, claim, or article for analysis."

    # Check for greetings and casual conversational prompts
    normalized_lower = cleaned_raw.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    if normalized_lower in GREETINGS_AND_CASUAL:
        return False, "Conversational greetings and casual queries are unsupported. Please enter a news headline, claim, or article for analysis."

    # Token-level checks
    words = re.findall(r"[a-zA-Z]+", cleaned_raw)
    if len(words) < 3:
        return False, "Input must contain at least 3 words to constitute a verifiable claim or headline. Please enter a news headline, claim, or article for analysis."

    # Check for repeated character gibberish (e.g. 'aaaaaaa', 'asdfghjkl')
    if re.search(r"(.)\1{4,}", cleaned_raw):
        return False, "Input contains excessive repeated characters. Please enter a news headline, claim, or article for analysis."

    # Check that at least 2 words have length >= 3
    substantive_words = [w for w in words if len(w) >= 3]
    if len(substantive_words) < 2:
        return False, "Input lacks substantive words. Please enter a news headline, claim, or article for analysis."

    return True, "Valid news text."
