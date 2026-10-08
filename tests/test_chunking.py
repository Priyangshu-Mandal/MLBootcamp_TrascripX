import re
import unittest

from pipeline.chunking import chunk_text, join_chunks


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


class ChunkingTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text("  \n\n ", 100), [])

    def test_small_text_single_chunk(self):
        ch = chunk_text("Hello there. General Kenobi.", 100)
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0].text, "Hello there. General Kenobi.")

    def test_paragraphs_packed_within_limit(self):
        text = "\n\n".join(f"Paragraph number {i} has some words." for i in range(10))
        ch = chunk_text(text, 100)
        self.assertGreater(len(ch), 1)
        self.assertTrue(all(len(c.text) <= 100 for c in ch))
        self.assertEqual(join_chunks([c.text for c in ch], ch), text)

    def test_long_paragraph_split_on_sentences_no_loss_no_duplication(self):
        text = " ".join(f"This is sentence number {i}." for i in range(60))
        ch = chunk_text(text, 200)
        self.assertGreater(len(ch), 3)
        self.assertTrue(all(len(c.text) <= 200 for c in ch))
        rebuilt = join_chunks([c.text for c in ch], ch)
        self.assertEqual(rebuilt, text)
        self.assertEqual(rebuilt.count("sentence number 7."), 1)

    def test_single_giant_sentence_hard_split(self):
        text = " ".join(f"word{i}" for i in range(500))
        ch = chunk_text(text, 120)
        self.assertTrue(all(len(c.text) <= 120 for c in ch))
        self.assertEqual(norm(join_chunks([c.text for c in ch], ch)), norm(text))

    def test_token_longer_than_limit(self):
        ch = chunk_text("a" * 500, 100)
        self.assertTrue(all(len(c.text) <= 100 for c in ch))
        self.assertEqual("".join(c.text for c in ch), "a" * 500)

    def test_paragraph_breaks_preserved_inside_chunk(self):
        text = "First para.\n\nSecond para."
        ch = chunk_text(text, 1000)
        self.assertEqual(ch[0].text, text)

    def test_join_length_mismatch(self):
        with self.assertRaises(ValueError):
            join_chunks(["a"], chunk_text("a b c d e f " * 20, 50))

    def test_tiny_limit_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", 10)


if __name__ == "__main__":
    unittest.main()
