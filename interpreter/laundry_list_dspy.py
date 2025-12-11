"""
Laundry list prompt challenge as a DSPy program.
Tests qwen3 8b on following multiple writing style rules.
"""

import re
import dspy

student_lm = dspy.LM("openrouter/qwen/qwen3-8b", cache=False)
dspy.configure(lm=student_lm)

INSTRUCTIONS = """
You are a writing assistant. Follow ALL of these rules exactly:

1. use lowercase for everything except proper names which must be ALL CAPS (like WILLIS TOWER, NAVY PIER)
2. never use bullet points or dashes for lists - use letter-numbered lists like a), b), c)
3. start every paragraph with the word "curiously,"
4. end every paragraph with exactly three dots (...)
5. never use the word "the" - always rephrase to avoid it
6. write exactly (4) paragraphs, no more, no less
7. mention exactly (3) chicago landmarks by name (no more, no less)
8. never use exclamation marks anywhere
9. include exactly (2) rhetorical questions (sentences ending with ?)
10. use the word "towering" exactly (2) times (not more, not less)
11. each paragraph should be exactly (4) sentences long
12. never use contractions (don't, can't, won't, etc.)
13. never use the word "city" - use alternatives like "metropolis" or "urban center"
14. include exactly one sentence that starts with "notably,"
15. the total response must be between 250 and 300 words
"""

class BlogWriter(dspy.Signature):
    """Write a blog post following all the style rules exactly."""
    topic: str = dspy.InputField(desc="The topic to write about")
    blog_post: str = dspy.OutputField(desc="The blog post following all style rules")

BlogWriterWithRules = BlogWriter.with_instructions(INSTRUCTIONS)
blog_writer = dspy.ChainOfThought(BlogWriterWithRules)


def evaluate_rules(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric: compute fraction of rules followed."""
    text = pred.blog_post
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    results = {}
    
    # Rule 1: proper names in ALL CAPS
    words = text.split()
    title_case_words = sum(1 for w in words if len(w) > 1 and w[0].isupper() and not w.isupper() and w[1:].islower())
    results["proper_names_all_caps"] = title_case_words < 3
    
    # Rule 2: no bullets
    has_bullets = any(line.strip().startswith(('-', '*', '•')) for line in text.split('\n'))
    results["no_bullets"] = not has_bullets
    
    # Rule 3: paragraphs start with "curiously,"
    curiously_count = sum(1 for p in paragraphs if p.lower().startswith('curiously,'))
    results["starts_with_curiously"] = curiously_count == len(paragraphs) if paragraphs else False
    
    # Rule 4: paragraphs end with ellipsis
    ellipsis_count = sum(1 for p in paragraphs if p.rstrip().endswith('...'))
    results["ends_with_ellipsis"] = ellipsis_count == len(paragraphs) if paragraphs else False
    
    # Rule 5: no "the"
    the_count = text.lower().split().count('the')
    results["no_the"] = the_count == 0
    
    # Rule 6: exactly 4 paragraphs
    results["exactly_4_paragraphs"] = len(paragraphs) == 4
    
    # Rule 7: exactly 3 landmarks
    landmarks = ["willis tower", "navy pier", "millennium park", "cloud gate", "bean", 
                 "john hancock", "wrigley field", "lake michigan", "chicago river", "magnificent mile",
                 "sears tower", "art institute", "buckingham fountain", "tribune tower", 
                 "water tower", "hancock center", "marina city", "wrigley building"]
    found_landmarks = [l for l in landmarks if l in text.lower()]
    results["exactly_3_landmarks"] = len(set(found_landmarks)) == 3
    
    # Rule 8: no exclamation marks
    results["no_exclamation"] = '!' not in text
    
    # Rule 9: exactly 2 rhetorical questions
    question_count = text.count('?')
    results["exactly_2_questions"] = question_count == 2
    
    # Rule 10: "towering" exactly 2 times
    towering_count = text.lower().count('towering')
    results["towering_exactly_2"] = towering_count == 2
    
    # Rule 11: each paragraph exactly 4 sentences
    para_sentence_counts = []
    for p in paragraphs:
        sentences = re.split(r'[.?!]+', p)
        sentences = [s.strip() for s in sentences if s.strip()]
        para_sentence_counts.append(len(sentences))
    valid_para_lengths = sum(1 for c in para_sentence_counts if c == 4)
    results["para_exactly_4_sentences"] = valid_para_lengths == len(paragraphs) if paragraphs else False
    
    # Rule 12: no contractions
    contractions = ["don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't", 
                    "hasn't", "haven't", "hadn't", "doesn't", "didn't", "couldn't", 
                    "wouldn't", "shouldn't", "it's", "that's", "there's", "here's"]
    has_contractions = any(c in text.lower() for c in contractions)
    results["no_contractions"] = not has_contractions
    
    # Rule 13: no "city"
    city_count = text.lower().split().count('city')
    city_count += text.lower().count("city'") + text.lower().count("city,") + text.lower().count("city.")
    results["no_city"] = city_count == 0
    
    # Rule 14: exactly one "notably," sentence
    sentences = re.split(r'[.?!]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    notably_starts = sum(1 for s in sentences if s.lower().startswith('notably,'))
    results["one_notably_sentence"] = notably_starts == 1
    
    # Rule 15: word count between 250-300
    word_count = len(text.split())
    results["word_count_250_300"] = 250 <= word_count <= 300
    
    # Return fraction of rules passed
    passed = sum(results.values())
    total = len(results)
    score = passed / total
    
    # For tracing/bootstrapping, require high compliance
    if trace is not None:
        return score >= 0.9
    
    return score


# Create examples (different topics to write about)
TOPICS = [
    "the skyline of chicago",
    "chicago architecture and its history", 
    "a walking tour of downtown chicago",
    "chicago landmarks at sunset",
    "exploring chicago by the lakefront",
]

devset = [
    dspy.Example(topic=topic).with_inputs("topic")
    for topic in TOPICS
]


if __name__ == "__main__":
    # Run evaluation
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=evaluate_rules,
        num_threads=4,
        display_progress=True,
        display_table=5,
    )
    
    print("Evaluating BlogWriter on laundry list rules...\n")
    result = evaluate(blog_writer)
    
    print(f"\n{'='*60}")
    print(f"Overall Score: {result.score:.1f}%")
    print(f"{'='*60}")
    
    # Show a sample output
    print("\nSample output:")
    sample = blog_writer(topic=TOPICS[0])
    print(sample.blog_post[:800] + "..." if len(sample.blog_post) > 800 else sample.blog_post)
