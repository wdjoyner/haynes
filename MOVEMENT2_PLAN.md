# Movement II — plan

What her Ph.D. work looked like. This plan revises a ChatGPT draft
(`AI_movement1_movement2.pdf`, not in the repo). The governing idea is that
Movement II shows **Haynes's actual mathematics**, not stand-ins.

Her dissertation (Catholic University of America, 1943):
*Determination of Sets of Independent Conditions Characterizing Certain
Special Cases of Symmetric Correspondences.* Each phrase of the title is a
visual:

- **Independent conditions**: each condition removes a degree of freedom.
  Movement I ended with freedom growing; Movement II opens by taking it away.
- **Symmetric correspondences**: a rule pairing points P ↔ Q on a line such
  that Q ↔ P as well. Plotted in a P–Q square, the graph is symmetric across
  the diagonal; points where P is its own partner are where the graph meets
  the diagonal (counted by Chasles's correspondence principle). Confirm with
  Edray Goins how her two representations work (see Sister C. F. Galvin's
  1938 CUA thesis, *Two geometrical representations of the symmetric
  correspondences C_{n,n}*) before narrating details.
- **Duality**: a curve as points vs. as its tangent lines; degree ↔ class,
  cusps ↔ flexes (Plücker numbers).

The Sage code by Goins and Joyner (`haynes-project-code_goins2.sage`, the
newer of two versions) implements the thesis curves — conic, cuspidal cubic,
trident, trinodal / bicuspidal / tricuspidal / lemniscate quartics, the
"Chapter N" examples — with their duals, Plücker invariants, tangent lines
from a point, and tangent-correspondence plots
(`plot_curve_correspondence_tangents`). Movement II scenes should be built on
those definitions.

## Running order (~6½ min)

| Scene | Content | Time |
|---|---|---|
| Bridge | Dissertation title written along a curve; its words fall into the first construction ("conditions" pins the first point) | 0:25 |
| 6 | Apollonius: grow the 8 tangent circles one at a time with a live counter; color by inside/outside contact (8 = 2×2×2). Called an analogy in narration, no on-screen disclaimer | 1:00 |
| 7 | Drag one circle across another: count changes 8 → 6 → 4 as solutions merge and vanish. Parameter plane colored by number of real solutions; a dot for the current configuration jumps count at region boundaries | 1:10 |
| 8 | Independent conditions: conic through 1, 2, … 5 points, the family visibly losing freedom until frozen. Slide a point until three are collinear: conic degenerates to two lines (conditions no longer independent) | 1:00 |
| 9 | Cubic family y² = x³ + ax + b: slide a coefficient through the nodal curve until it splits into two pieces. Sweep a line across it: visible intersections 3 → tangency → 1, counter still reads 3 (complex solutions, multiplicity) | 1:00 |
| 10 | Symmetric correspondence: P sliding on a line, partners linked by arcs; the P–Q graph mirrored across the diagonal; coincidences flash and are counted | 1:00 |
| 11 | Duality: tangent line rolling around the tricuspidal quartic (deltoid) while the dual curve draws itself; Plücker counts stay fixed as the curve deforms. Words RELATION → INVARIANCE → STRUCTURE on curves; her 1945 line on invariance, "that which remains stable in midst of change" | 0:50 |
| Closing | Simson lines of the Movement I 3-4-5 triangle sweep out a deltoid (Steiner) — classroom geometry becoming thesis geometry. Then regular polygons closing in on a circle (her 1945 "limit … goal of perfection" example). End with "We will teach, we will do more than that, we will inspire," INSPIRE breaking free of its curve | 0:45 |

Cut from the draft: the rotating Platonic solid (unrelated to her work; and
it invites reading "symmetric correspondences" as polyhedral symmetry).

## Continuity with Movement I

- Reuse `CurveText` / `write_along` (words written along and riding curves),
  `SwingingDoor`, the CHALK color, and the narration-manifest timing.
- Put all words in a `haynes_movement2_text.json`, mirroring Movement I.
- "We will teach … inspire" currently closes Movement I's credits; move it to
  Movement II's ending, and give Movement I a different closing line
  (candidates: the invariance or the limit passage from the 1945 address).
- Draft title "Mathematics as Discovery" vs. film title "Mathematics as an
  Open Door": decide whether "Discovery" is Movement II's subtitle.

## Sources for narration

Haynes, "Mathematics — Symbolic Logic," address to teachers of mathematics,
Washington, D.C., 1945 (invariance, limit, construction of concepts,
"possible worlds" passages). Kelly, Shinners, Zoroufy, Notices of the AMS 64
(2017).
