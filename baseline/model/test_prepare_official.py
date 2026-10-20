import unittest
from prepare_official import clean_body, UnionFind


class CleaningTests(unittest.TestCase):
    def test_caption_accents_quotes_and_repeated_prefix(self):
        record = {'caption': 'André works at “New York”.', 'headline': 'A long headline about artists',
                  'article': 'Andre works at "New York".\nCredit...Photographer\n\nAndre works at "New York".\nA long headline about artists\nActual news starts here.'}
        text, actions = clean_body(record)
        self.assertEqual(text, 'Actual news starts here.')
        self.assertEqual(actions, ['leading_caption', 'caption_credit', 'leading_caption', 'leading_headline'])

    def test_interior_caption_and_partial_prefix_preserved(self):
        caption = 'This is a full caption.'
        for text in ('An introduction. ' + caption, 'This is a full but different sentence.'):
            self.assertEqual(clean_body({'article': text, 'caption': caption})[0], text)

    def test_short_metadata_not_removed_and_entities_decoded(self):
        text, actions = clean_body({'article': 'Good news &amp; updates.', 'headline': 'Good news'})
        self.assertEqual(text, 'Good news & updates.')
        self.assertEqual(actions, [])

    def test_duplicate_identity_chains(self):
        groups = UnionFind()
        for _ in range(4):
            groups.add()
        groups.union(0, 1)
        groups.union(2, 3)
        groups.union(1, 3)
        self.assertEqual(len({groups.find(i) for i in range(4)}), 1)


if __name__ == '__main__':
    unittest.main()
