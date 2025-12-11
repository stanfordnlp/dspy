"""
Laundry list prompt challenge: Find a set of rules hard enough that qwen3 8b 
only follows ~60% of them correctly.
"""

import dspy

student_lm = dspy.LM("openrouter/qwen/qwen3-8b", cache=False)
dspy.configure(lm=student_lm)

RULES = """
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

def evaluate_response(response: str) -> dict:
    """Check which rules are followed."""
    import re
    text = response
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    results = {}
    
    # Rule 1: lowercase except proper names in ALL CAPS
    # Check that regular words are lowercase, proper names are all caps
    words = text.split()
    # Count words that are title case (first letter cap, rest lower) - these should be ALL CAPS instead
    title_case_words = sum(1 for w in words if w[0].isupper() and not w.isupper() and len(w) > 1 and w[1:].islower())
    results["proper_names_all_caps"] = title_case_words < 3
    
    # Rule 2: no bullets, uses letter lists
    has_bullets = any(line.strip().startswith(('-', '*', '•')) for line in text.split('\n'))
    results["no_bullets"] = not has_bullets
    
    # Rule 3: paragraphs start with "curiously,"
    curiously_count = sum(1 for p in paragraphs if p.lower().startswith('curiously,'))
    results["starts_with_curiously"] = curiously_count == len(paragraphs) if paragraphs else False
    
    # Rule 4: paragraphs end with ellipsis
    ellipsis_count = sum(1 for p in paragraphs if p.rstrip().endswith('...'))
    results["ends_with_ellipsis"] = ellipsis_count == len(paragraphs) if paragraphs else False
    
    # Rule 5: no "the" at all
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
    # Also check for "city's" and "city,"
    city_count += text.lower().count("city'") + text.lower().count("city,") + text.lower().count("city.")
    results["no_city"] = city_count == 0
    
    # Rule 14: exactly one sentence starting with "notably,"
    sentences = re.split(r'[.?!]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    notably_starts = sum(1 for s in sentences if s.lower().startswith('notably,'))
    results["one_notably_sentence"] = notably_starts == 1
    
    # Rule 15: word count between 250-300
    word_count = len(text.split())
    results["word_count_250_300"] = 250 <= word_count <= 300
    
    return results

def run_test(topic: str, n_trials: int = 5):
    """Run multiple trials and report average compliance."""
    
    prompt = f"""{RULES}

Topic: {topic}

Write the blog post now, following ALL rules above:"""
    
    all_results = []
    for i in range(n_trials):
        raw_response = student_lm(prompt, temperature=0.7)[0]
        # Handle dict response (qwen3 with reasoning)
        if isinstance(raw_response, dict):
            response = raw_response.get('text', str(raw_response))
        else:
            response = str(raw_response)
        print(f"\n{'='*60}")
        print(f"Trial {i+1}:")
        print(f"{'='*60}")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        results = evaluate_response(response)
        all_results.append(results)
        
        passed = sum(results.values())
        total = len(results)
        print(f"\nRules passed: {passed}/{total} ({100*passed/total:.1f}%)")
        for rule, passed in results.items():
            print(f"  {'✓' if passed else '✗'} {rule}")
    
    # Average across trials
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL TRIALS")
    print(f"{'='*60}")
    
    rule_names = list(all_results[0].keys())
    for rule in rule_names:
        pass_rate = sum(r[rule] for r in all_results) / n_trials
        print(f"{rule}: {pass_rate*100:.0f}%")
    
    overall = sum(sum(r.values()) for r in all_results) / (n_trials * len(rule_names))
    print(f"\nOVERALL COMPLIANCE: {overall*100:.1f}%")
    return overall

if __name__ == "__main__":
    topic = "write a blog post about chicago skyline"
    compliance = run_test(topic, n_trials=5)
    print(f"\nTarget was ~60%, achieved {compliance*100:.1f}%")
