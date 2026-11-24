# Question pairs for the Arabic translation benchmark

`translation_pairs.csv` holds 30 questions that people might send the bot. Each
one is written twice: in Hebrew, and in spoken Palestinian Arabic.

The bot's information is in Hebrew. When someone writes in Arabic, the bot first
translates the question into Hebrew and then searches. We use these pairs to
test which translator does that best: for each pair, the Arabic question should
find the same information as the Hebrew one.

The Arabic column is a **first draft and needs checking by a native speaker.**

## What to check in each row

1. **Is the Arabic how people really write?** Picture someone typing on WhatsApp,
   not an official letter: everyday Palestinian Arabic, the words people
   actually use. Rewrite anything that sounds like a translation or like formal
   Arabic (فصحى).
2. **Does it mean the same as the Hebrew?** Same question, same details. If the
   Arabic is better as it is and the Hebrew is a little off, fix the Hebrew
   instead and say so in `notes`.
3. **Hebrew words in Arabic.** If people would usually use a Hebrew word in the
   middle of the Arabic (for example بيطوح لئومي instead of التأمين الوطني),
   write it the way they would, and mention it in `notes`.

When a row is done, put your name in `reviewed_by`.

## Adding questions

New rows are welcome, especially ways of asking that people really use and that
are missing here (spelling mistakes are fine, they're realistic). Give each new
row a new `id` (q31, q32, ...). It's also fine to add a second Arabic version of
an existing question as its own row, for example `q02b` with the same Hebrew.

Questions in Arabic written with Latin letters ("shu 7u2u2i") don't work with
this test yet. If people often write that way, write that down in `notes`.

## Saving the file

Open it in Excel or Google Sheets, but **save it as CSV UTF-8** so the Arabic
and Hebrew stay readable. Keep the column names in the first row as they are.

---

For developers: the benchmark that reads this file is
`scripts/translation_bench.py`. See its docstring for how to run it and what the
columns of its report mean.
