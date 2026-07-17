# Model Comparison

## Results

| Model | Final Val Loss | Byte-Level Perplexity |
|---|---:|---:|
| A | 2.0381 | 7.6761 |
| B | 1.8026 | 6.0652 |

Note: byte-level perplexity is only meaningful for comparing Model A vs Model B in this project.


| Prompt | Model | Structural stability | Shakespearean styling | Repetition loops? |
|---|---|---|---|---|
| Come, cousin, canst thou quake, | Model A | **Low** - malformed words and a broken second line undermine continuity. | **Low** - “thould” and “gonst” hint at archaic diction, but the phrasing is not convincingly Shakespearean. | **Minor** - repeats “the” and “his,” but no fragment dominates the full continuation. |
| Come, cousin, canst thou quake, | Model B | **Medium** - mostly recognizable words and phrases, though the thought remains incomplete. | **Medium** - dramatic line breaks and words such as “heir” and “grace” suggest the genre inconsistently. | **Minor** - “to me” repeats near the start, without becoming a sustained loop. |
| Come, cousin, canst thou quake, | Gemini Flash | **High** - a complete, grammatical question continues naturally across two balanced lines. | **High** - “pale thy cheek” and the storm image sustain a clear Elizabethan dramatic voice. | **No obvious loop** - the continuation develops one image without repeated fragments. |
| O Romeo, Romeo! wherefore art thou | Model A | **Low** - malformed words and abrupt phrase changes prevent a coherent sentence. | **Low** - the prompt supplies most of the Shakespearean signal; the continuation lacks stable dramatic phrasing. | **Yes** - “my my my” is an obvious local repetition loop. |
| O Romeo, Romeo! wherefore art thou | Model B | **Medium** - readable clauses and a speaker label appear, but syntax and meaning remain weak. | **Medium** - verse layout and “GLOUCESTER” imitate drama, though diction is inconsistent. | **Minor** - “the” and “to” recur noticeably, but do not dominate the continuation. |
| O Romeo, Romeo! wherefore art thou | Gemini Flash | **High** - coherent, grammatical lines complete the thought cleanly. | **High** - the response preserves Juliet's dramatic cadence and genre-appropriate diction. | **No obvious loop** - repeated “Romeo” is purposeful source phrasing, not degeneration. |
| Now is the winter of our discontent | Model A | **Low** - broken clauses, invented words, and an abrupt speaker change disrupt structure. | **Low** - a play-like label appears, but the surrounding language does not sustain the style. | **No obvious loop** - the output drifts without heavily repeating a phrase. |
| Now is the winter of our discontent | Model B | **Medium** - line and speaker structure are stable, but sentence continuity is weak. | **Medium** - “gracious,” “thee,” and dialogue formatting evoke Shakespeare unevenly. | **Minor** - “by” repeats locally, but no phrase fragment dominates. |
| Now is the winter of our discontent | Gemini Flash | **High** - a fluent two-line continuation maintains grammar and a unified image. | **High** - metaphor, rhythm, and elevated diction strongly fit the genre. | **No obvious loop** - the image progresses without repeated wording. |
| To be or none or little; | Model A | **Low** - several malformed words and truncated speaker turns make the passage unstable. | **Low** - speaker labels suggest drama, but the continuation has little credible Elizabethan phrasing. | **No obvious loop** - repeated roots such as “com-” are noticeable, but not a dominating phrase loop. |
| To be or none or little; | Model B | **Medium** - mostly readable clauses and stable line breaks, despite malformed words and an unfinished ending. | **Medium** - “gonest,” “thee,” and the speaker label create intermittent Shakespearean flavor. | **No obvious loop** - the continuation changes phrases rather than cycling through one fragment. |
| To be or none or little; | Gemini Flash | **High** - a coherent grammatical statement develops the prompt into a complete idea. | **High** - “'tis,” “mortal crown,” and the measured cadence sustain an Elizabethan tone. | **No obvious loop** - no phrase or short-word cycle appears. |

### Production-model method

The production baseline is **Gemini Flash**, evaluated on the four lines in `evaluation/prompts.txt`. It was asked to continue each line in Shakespearean style, preserve the original line, and limit each continuation to 150 characters so that its output length was comparable to Models A and B. The resulting text is saved in `evaluation/outputs/gemini_flash.txt`.
