import  re
import os

def second_cleanup(md_doc):
    cleaned=re.sub(r"\n\s*\n",'\n\n',md_doc)
    cleaned=cleaned.strip()
    exclude_phrases=["all rights reserved"," privacy policy","SISFSYou need to enable JavaScript to run this app.",
        "Loading..."]
    for phrase in exclude_phrases:
        cleaned=cleaned.replace(phrase,'')
    return cleaned;

input_folder='data'
out_folder="data/cleaned"
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