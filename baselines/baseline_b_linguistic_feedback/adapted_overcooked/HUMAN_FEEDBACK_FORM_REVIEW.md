# Human feedback-form data review

The JSON syntax error in the tomato instruction has been fixed. Blank lines are
ignored. The 50 remaining rows all satisfy the two-field schema.

The following semantic edits were **not** made automatically because they would
change the annotator's intended meaning:

- Rows 1-3 duplicate the documentation example. They are valid labels but must
  not be described as naturally elicited player language unless that is true.
- Row 44 (`Go to deliver the soup ...`) has a broken object/reason relation and
  should be rewritten or removed after checking the intended sentence.
- Rows 27 and 35 are general behavior instructions. They are reasonable speech
  acts, but they do not cleanly match the paper's narrow `action_spatial ->
  Imperative` collapse.
- Rows 47 and 48 can be read either as indirect requests or state/value
  descriptions. Keep `Descriptive` only if that was the intended reading.

Spelling mistakes are retained as raw human language. Corrected variants are
added only to the train-only augmentation file.

Current collection quality: Evaluative has only two automatically derived
families and both share the same basic surface construction. Add several
substantially different Evaluative phrasings
before treating a future human test score as a generalization result.
