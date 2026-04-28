"""
Translation Accuracy Testing and Improvement Demonstration
Tests the enhanced translation module with Latin text samples
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.translation import TextTranslator


# Test dataset: (Latin text, expected English meaning)
TEST_CASES = [
    ("Salve, munde!", "Hello, world!"),
    ("Amor vincit omnia.", "Love conquers all."),
    ("Carpe diem.", "Seize the day."),
    ("Veni, vidi, vici.", "I came, I saw, I conquered."),
    ("In principio erat Verbum.", "In the beginning was the Word."),
    ("Dominus vobiscum.", "The Lord be with you."),
    ("Deus vult.", "God wills it."),
    ("Tempus fugit.", "Time flees."),
    ("Memento mori.", "Remember you must die."),
    ("Fides quaerens intellectum.", "Faith seeking understanding."),
    ("Et cetera.", "And the rest."),
    ("Quid pro quo.", "Something for something."),
    ("De facto.", "In fact."),
    ("Vita est brevis.", "Life is short."),
    ("Homines naturali ratione instructi sunt.", "Men are endowed with natural reason."),
]

# Test cases with diacritical marks and abbreviations
CHALLENGING_CASES = [
    ("Sānctus Augustīnus dixit.", "Saint Augustine said."),
    ("Pp. et ff. in via.", "Fathers and brothers on the road."),
    ("Sc. hominibus.", "Namely to men."),
    ("Dd. et Mpp.", "God and martyrs."),
]


def test_translation_accuracy():
    """Test translation accuracy and display metrics"""
    print("="*70)
    print("LATIN TRANSLATION ACCURACY TEST")
    print("="*70)
    print()
    
    translator = TextTranslator(source_lang='la', target_lang='en')
    
    all_results = []
    
    # Test standard cases
    print("STANDARD TEST CASES:")
    print("-" * 70)
    
    for latin_text, expected_meaning in TEST_CASES:
        result = translator.translate_text(latin_text)
        
        print(f"\nLatin:         {latin_text}")
        print(f"Translation:   {result['translated_text']}")
        print(f"Confidence:    {result.get('confidence', 0.0):.2%}")
        print(f"Validation:    {result.get('validation_score', 0.0):.2%}")
        print(f"Status:        {'✓ Translated' if result['translated'] else '✗ Failed'}")
        
        all_results.append(result)
    
    print("\n" + "="*70)
    print("CHALLENGING CASES (with diacritics and abbreviations):")
    print("-" * 70)
    
    for latin_text, expected_meaning in CHALLENGING_CASES:
        result = translator.translate_text(latin_text)
        
        print(f"\nLatin:         {latin_text}")
        print(f"Translation:   {result['translated_text']}")
        print(f"Confidence:    {result.get('confidence', 0.0):.2%}")
        print(f"Validation:    {result.get('validation_score', 0.0):.2%}")
        print(f"Status:        {'✓ Translated' if result['translated'] else '✗ Failed'}")
        
        all_results.append(result)
    
    # Calculate statistics
    print("\n" + "="*70)
    print("ACCURACY STATISTICS:")
    print("-" * 70)
    
    stats = translator.get_translation_stats(all_results)
    
    print(f"\nTotal texts processed:    {stats['total']}")
    print(f"Successful translations:  {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%})")
    print(f"Failed translations:      {stats['failed']}")
    print(f"Average confidence score: {stats['avg_confidence']:.2%}")
    print(f"Average validation score: {stats['avg_validation_score']:.2%}")
    
    # Confidence distribution
    confidences = [r.get('confidence', 0.0) for r in all_results if r.get('translated')]
    if confidences:
        min_conf = min(confidences)
        max_conf = max(confidences)
        avg_conf = sum(confidences) / len(confidences)
        
        print(f"\nConfidence range:")
        print(f"  Minimum: {min_conf:.2%}")
        print(f"  Maximum: {max_conf:.2%}")
        print(f"  Average: {avg_conf:.2%}")
    
    print("\n" + "="*70)
    print("ENHANCEMENT FEATURES USED:")
    print("-" * 70)
    print("✓ Unicode normalization (NFKC)")
    print("✓ Diacritic removal for better model compatibility")
    print("✓ Abbreviation expansion (et → et cetera, pp → patres, etc.)")
    print("✓ Sentence-aware text chunking")
    print("✓ Latin word validation")
    print("✓ Beam search (num_beams=4) for better translation quality")
    print("✓ Temperature control (0.7) for more consistent output")
    print("✓ Repetition penalty (1.2) to avoid repetitive translations")
    print("✓ Translation caching for repeated texts")
    print("✓ Confidence scoring based on multiple factors")
    print()


def test_batch_processing():
    """Test batch translation processing"""
    print("\n" + "="*70)
    print("BATCH PROCESSING TEST")
    print("="*70)
    print()
    
    translator = TextTranslator(source_lang='la', target_lang='en')
    
    batch_texts = [text for text, _ in TEST_CASES[:5]]
    
    print("Processing batch of 5 Latin texts...")
    results = translator.batch_translate_with_batching(batch_texts, batch_size=2)
    
    print(f"Processed {len(results)} texts")
    
    stats = translator.get_translation_stats(results)
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Average confidence: {stats['avg_confidence']:.2%}")
    print()


def test_caching():
    """Test translation caching"""
    print("\n" + "="*70)
    print("TRANSLATION CACHING TEST")
    print("="*70)
    print()
    
    translator = TextTranslator(source_lang='la', target_lang='en')
    
    text = "Amor vincit omnia."
    
    print(f"Translating: '{text}'")
    print("First call (from model)...")
    result1 = translator.translate_text(text)
    print(f"  Result: {result1['translated_text']}")
    print(f"  Cache size: {len(translator.translation_cache)}")
    
    print("\nSecond call (from cache)...")
    result2 = translator.translate_text(text)
    print(f"  Result: {result2['translated_text']}")
    print(f"  Cache size: {len(translator.translation_cache)}")
    print(f"  Results identical: {result1['translated_text'] == result2['translated_text']}")
    print()


if __name__ == "__main__":
    try:
        test_translation_accuracy()
        test_batch_processing()
        test_caching()
        
        print("\n" + "="*70)
        print("✓ All tests completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
