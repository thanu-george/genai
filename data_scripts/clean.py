import  re
import os

import re

def second_cleanup(md_doc):
    # normalize line endings and collapse long blank runs
    cleaned = re.sub(r"\r\n", "\n", md_doc)
    cleaned = re.sub(r"\n[ \t]*\n+", "\n\n", cleaned)

    # common phrase-level removals (literal)
    exclude_phrases = [
        "all rights reserved",
        "privacy policy",
        "[Jump to navigation]",
        "(#main-menu)",
        "* [Previous](#)",
        "* [Next](#)",
        "Pause","###Click here to know more"
    ]
    for phrase in exclude_phrases:
        cleaned = cleaned.replace(phrase, "")

    # remove close buttons: ×, X (isolated), [×](#)
    cleaned = re.sub(r"\[×\]\(#\)|×", "", cleaned)

    # remove standalone UI junk lines
    cleaned = re.sub(r"^\s*(Ok|Yes|No|Submit|SUBMIT)\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)

    # remove repetitive "Click here to know more" headers
    cleaned = re.sub(r"###\s*Click here to know more\s*", "", cleaned, flags=re.IGNORECASE)

    # remove entire modal/dialog sections starting with ### headers (case-insensitive)
    modal_headers = [
        "Log in to",
        "Please Login/Register",
        "Error",
        "Do you really want to logout",
        "Notification Alert",
        "Your password must contain",
        "Please Complete Your Profile",
        "Your profile is currently under moderation",
    ]
    for header in modal_headers:
        cleaned = re.sub(
            rf"(?im)^\s*###\s*{re.escape(header)}\b.*?(?=(?:\n\s*###\s|\Z))",
            "",
            cleaned,
            flags=re.DOTALL,
        )

    # remove lines that are only a star or a star+spaces
    cleaned = re.sub(r"^\*\s*$", "", cleaned, flags=re.MULTILINE)

    # normalize bullet-with-link indentation: "   * [Link]" -> "* [Link]"
    cleaned = re.sub(r"(?m)^[ \t]*\*[ \t]*", "* ", cleaned)

    # remove numbered junk lists like "1. 1", "2. 2", etc.
    cleaned = re.sub(r"(?m)^[ \t]*\d+\.\s*\d+\s*$", "", cleaned)
    
    # normalize headings: strip leading spaces before # signs
    cleaned = re.sub(r"(?m)^[ \t]+(?=#)", "", cleaned)

    # remove lines (or long inline sequences) that include common navigation/control tokens
    cleaned = re.sub(
    r"\[Go to Previous\].*?Play Light Show.*?\)",
    "",
    cleaned,
    flags=re.DOTALL,
    )
    # remove empty/broken JS links
    cleaned = re.sub(r"\[\]\(javascript:void\(0\)\)", "", cleaned, flags=re.IGNORECASE)

    # remove lines that are just ---
    cleaned = re.sub(r"^---\s*$", "", cleaned, flags=re.MULTILINE)

    # remove nav menus (HOME, ABOUT, etc.)
    cleaned = re.sub(
    r"(?:\[HOME.*?\]\(.*?\)\s*){2,}",  # collapses repeated nav links
    "",
    cleaned,
    flags=re.DOTALL | re.IGNORECASE,
    )

    # remove breadcrumb-style menus with »
    cleaned = re.sub(r"\[.*?»\]\(.*?\)", "", cleaned)

    # remove leftover empty lines and trim
    cleaned = re.sub(r"\n[ \t]+\n", "\n\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


input_folder='../data/first_clean'
out_folder="../data/second_clean"
os.makedirs(out_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.endswith(".md"):
        input_path=os.path.join(input_folder,fname)
        out_path=os.path.join(out_folder,fname)
        with open(input_path,'r') as f:
            content=f.read()
        clean_content=second_cleanup(content)
        with open(out_path,'w') as f:
            f.write(clean_content)