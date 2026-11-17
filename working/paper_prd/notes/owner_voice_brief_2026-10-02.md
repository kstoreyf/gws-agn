FINAL AUTHORIAL REWRITE + FIGURE DESIGN PASS

Work on the current `working/paper_codex/`.

Read the CURRENT compiled PDF completely before touching anything.

This is not another incremental copy-edit.

The scientific results, likelihood, numbers, and broad section structure are correct. The remaining problem is that the manuscript still reads like a polished internal research note: it repeatedly describes simulations as experiments, explains which quantity was held fixed, narrates which calculation was run next, and tells the reader what each test was designed to isolate.

The figures have the same problem: several still look like analysis plots, with legend strips sitting above panels and visual encodings designed for interactive inspection rather than a journal page.

Fix both problems together.

DO NOT add new science.
DO NOT change numerical results.
DO NOT resurrect old estimator comparisons.
DO NOT add debugging discussion.
DO NOT commit or push.

The target is a finished astrophysics paper.

---

# 1. CHANGE THE AUTHORIAL POINT OF VIEW

The paper must stop sounding like:

> We built an experiment.
> We changed one assumption.
> We ran another experiment.
> We held this mock fixed.
> This calculation was designed to isolate X.
> Figure Y shows the response.
> This result tells us Z.

Instead it should read like a normal astrophysics paper:

> Here is the population model.
> Here is the observable.
> Here is the likelihood.
> Here is what the posterior says.
> Here is the physical reason.
> Here is where the inference becomes limited.

The reader does NOT need to experience the chronology of the project.

The final paper should conceal the history of how the authors arrived at the final analysis, except where a methodological distinction is scientifically necessary.

---

# 2. BAN “EXPERIMENT REPORT” PROSE

Search the entire manuscript and rewrite sentences with constructions like:

- “We study this problem…”
- “The simulations are designed to…”
- “This choice isolates…”
- “To isolate the effect…”
- “We next…”
- “We now…”
- “The calculation uses…”
- “The same calculation…”
- “This calculation reveals…”
- “We repeat the inference…”
- “The response is measured by…”
- “For the same simulated catalog…”
- “The fixed simulated catalog studied here…”
- “This gives a conditional sensitivity…”
- “These curves hold one simulated catalog fixed…”
- “The present calculation…”
- “At this event count and depth…”
- “The experiment…”
- “the simulated catalog considered”
- “the catalog used in the depth study”
- “the calculation in Figure X”
- “this test isolates”
- “this result identifies”

Some of these ideas are scientifically necessary. The wording is not.

State the physical/statistical condition instead.

---

# 3. CONCRETE EXAMPLES FROM THE CURRENT DRAFT

These are exactly the kinds of sentences that must disappear.

## Current:

> “The specific contribution is to quantify this multitracer inference when the catalogs are incomplete, their abundance normalizations are uncertain, or those normalizations are inferred with the population parameters.”

This sounds like a grant abstract.

Rewrite around the physical problem:

> “Catalog incompleteness couples the inferred host fraction to the abundance of sources below the survey limit, introducing an additional degeneracy with the sparse AGN population.”

Then state what the paper finds.

---

## Current:

> “We study this problem with simulated gravitational-wave observations and galaxy and AGN catalogs drawn from a common large-scale density field.”

Generic AI setup sentence.

Replace with something concrete:

> “We construct galaxy and AGN catalogs from the same large-scale density field and assign binary mergers to the two populations with probability \(f_{\rm AGN}\).”

Or fold this directly into the next sentence.

---

## Current:

> “The simulations are designed to isolate the information that distinguishes two host populations.”

Delete.

Nothing is gained.

Begin Section 3 with the actual mock definition.

For example:

> “The mock catalogs contain a dense galaxy population and a sparse, more strongly biased AGN population drawn from a common lognormal density field.”

That is astrophysics.

---

## Current:

> “This choice holds their radial selection approximately fixed, separating the effect of abundance and clustering from differences in the luminosity function.”

This can be shorter and more natural:

> “We use the same luminosity function for both tracers, so their differences arise primarily from abundance and clustering bias.”

---

## Current:

> “To isolate the effect of depth and the assumed AGN completion normalization, the flux threshold is changed while the underlying events and large-scale structure are held fixed.”

Too much internal-experiment voice.

Write:

> “We evaluate four flux limits, corresponding to integrated completeness values of 99.7%, 81.4%, 31.5%, and 9.5%.”

Then later, if essential:

> “The same merger catalog is used at each flux limit.”

One sentence. No “isolate.”

---

## Current:

> “The same simulated gravitational-wave events and large-scale structure are used at every threshold, isolating the response to catalog depth.”

Again, delete the meta-clause.

> “We use the same merger catalog at each threshold.”

Done.

---

## Current caption language:

> “the figure gives a conditional sensitivity for this simulated catalog.”

Do not write captions like a methods memo.

Just say:

> “All curves use the same merger catalog.”

The reader understands what that means.

---

## Current Discussion:

> “The depth and density-response curves hold one simulated catalog fixed and do not determine an ensemble bias or a general survey requirement.”

This is referee-response prose.

Replace with the actual scientific limitation:

> “The magnitude of these shifts will depend on the tracer density, clustering bias, survey selection, and GW localization distribution.”

That is enough.

---

## Current:

> “Those ingredients are needed to forecast how quickly real data can constrain the environmental fraction, although they do not alter the identifiability questions isolated here.”

Very AI.

Prefer:

> “Forecasts for LVK observations will additionally require detector duty cycles, realistic sky maps, waveform-dependent selection, and the full mass–redshift population.”

Stop there.

---

# 4. REMOVE META-SCIENCE WORDS

Strongly reduce or eliminate:

- isolate
- identifiability, unless mathematically necessary
- sensitivity, when “dependence” or an actual derivative is clearer
- conditional
- response, when “change” or derivative is clearer
- configuration
- scenario
- setup
- experiment
- calculation, repeated constantly
- mock family
- tested
- behavior
- framework
- prescription
- regime
- information content
- environmental inference
- cosmological inference, when simply `H0` is clearer
- observational application, repeated constantly

Do not simply substitute synonyms.

Rewrite around the quantity itself.

Instead of:

> “The cosmological inference remains compatible…”

say:

> “The \(H_0\) posterior contains the injected value…”

Instead of:

> “the environmental inference broadens…”

say:

> “The \(f_{\rm AGN}\) posterior broadens…”

Specific nouns beat abstract AI nouns.

---

# 5. STOP CALLING EVERYTHING “INFERENCE”

There are too many sentences of the form:

- joint inference
- cosmological inference
- environmental inference
- multitracer inference
- fixed-normalization inference
- free-normalization inference

Use the actual parameter or posterior.

Examples:

BAD:
> “The joint inference remains consistent…”

GOOD:
> “The \(H_0\) and \(f_{\rm AGN}\) posteriors contain the injected values…”

BAD:
> “freeing the normalizations broadens the environmental inference”

GOOD:
> “Marginalizing over \(n_{0,\rm gal}\) and \(n_{0,\rm AGN}\) broadens the \(f_{\rm AGN}\) posterior…”

---

# 6. INTRODUCTION MUST SOUND LIKE A PAPER, NOT A PROJECT DESCRIPTION

The first three paragraphs are mostly good.

The weak part begins around:

> “Here every event may arise…”
> “The specific contribution is…”
> “We study this problem…”

Rewrite the final Introduction paragraphs from scratch.

Do not describe the work as a list of things we quantify.

The end of the Introduction should state the physical issue and principal results.

The logic should be approximately:

1. GW–AGN association and dark-siren cosmology use the same spatial/redshift information.
2. A two-tracer population model therefore allows \(H_0\) and \(f_{\rm AGN}\) to be inferred simultaneously.
3. Missing hosts introduce an additional dependence on the abundance of each tracer below the survey limit.
4. In the mocks here, \(H_0\) and \(f_{\rm AGN}\) are weakly correlated when the catalogs are complete.
5. The principal limitation on \(f_{\rm AGN}\) is uncertainty in the sparse AGN normalization.
6. That uncertainty begins to affect \(H_0\) for shallow catalogs.
7. Marginalizing over the tracer normalizations constrains the dense galaxy normalization much more strongly than the sparse AGN normalization.

Do not say “we study,” “we investigate,” “we quantify,” “we explore,” or “we test” in every sentence.

---

# 7. METHODS SHOULD BE MATHEMATICAL, NOT PROCEDURAL

Section 2 currently still contains procedural details like:

> “The calculation uses 2,205,380 detected injections from 150,000,000 draws.”

This may be useful in an appendix, but it interrupts the main statistical model.

Main Methods should contain:

- hierarchical likelihood;
- host mixture;
- observed and missing-host intensities;
- selection normalization;
- free parameters and priors.

Move large Monte Carlo counts, proposal-component weights, and detailed effective-sample diagnostics to Appendix B unless they are needed to understand an equation.

Section 2.3 should be considerably shorter.

The main paper is not about how many injections the code generated.

---

# 8. SECTION 3 SHOULD READ LIKE “MOCK CATALOGS,” NOT “SIMULATED EXPERIMENT”

Consider renaming:

`3. SIMULATED OBSERVATIONS`

to something more conventional such as:

`3. MOCK CATALOGS AND GRAVITATIONAL-WAVE EVENTS`

or

`3. MOCK GALAXY, AGN, AND GRAVITATIONAL-WAVE CATALOGS`

Choose the shortest natural option.

Do not begin the section by explaining what the simulation was “designed” to do.

Begin with the objects and distributions.

Similarly, consider renaming:

`3.3 Catalog selection`

only if another title is more natural, e.g.

`3.3 Flux-limited catalogs`

which directly describes the data product.

---

# 9. RESULTS SHOULD OPEN WITH RESULTS

Section 4 currently begins:

> “We begin with complete galaxy and AGN catalogs and infer only…”

This is still analysis-sequence language.

Prefer:

> “For complete galaxy and AGN catalogs, the \(H_0\) and \(f_{\rm AGN}\) posteriors are only weakly correlated.”

Then give Figure 2 and the numerical ensemble result.

Likewise Section 5 should not begin:

> “We now apply the flux limits…”

Start with the result:

> “Reducing the integrated completeness from 100% to 9.5% produces only a modest increase in the \(H_0\) and \(f_{\rm AGN}\) credible intervals when the tracer normalizations are fixed.”

Then explain the magnitude thresholds.

Likewise Section 6 should not begin:

> “We next infer…”

Start with the scientific statement:

> “Marginalizing over the two completion-density normalizations has very different consequences for the galaxy and AGN components.”

Then report what happens.

---

# 10. SECTION TITLES SHOULD NOT SOUND LIKE ANALYSIS TASKS

Review subsection titles.

Current examples like:

`Response to the AGN completion normalization`

still sound experimental.

Consider:

`Dependence on the AGN abundance`

or

`AGN abundance and the host fraction`

depending on the precise quantity.

Likewise:

`Completion-normalization constraints`

could become:

`Galaxy and AGN abundance`

or another simple physical title.

Avoid over-engineered titles.

---

# 11. DISCUSSION NEEDS THE BIGGEST VOICE CLEANUP

The Discussion still sounds like us talking through the experiment.

Rewrite it more substantially than the other sections.

It should read as astrophysical interpretation.

For example, Section 7.1 should not begin:

> “The complete-catalog calculations show that…”

Just state:

> “When the galaxy and AGN intensity fields are known, \(H_0\) and \(f_{\rm AGN}\) affect the likelihood in different ways.”

Then explain why.

Current:

> “This measurement is closely related to…”

Prefer:

> “The host-mixture term in Equation (3) is closely related to statistical GW–AGN association analyses…”

Current:

> “The single-tracer calculation in Appendix C gives…”

Prefer:

> “The sparse AGN catalog provides substantially more redshift information per associated event in these mocks…”

Then cite Appendix C parenthetically if useful.

Do not narrate where in the paper the supporting result lives unless necessary.

---

# 12. LIMITATIONS SHOULD NOT READ LIKE A DEFENSE

The current limitations section contains several sentences that sound written for an imaginary referee.

Keep the actual limitations, but state them plainly.

Good limitations:

- AGN and galaxies are mutually exclusive in the mock;
- real AGN are a subset of galaxies;
- mass/spin populations are fixed;
- radial selection-function shapes are fixed;
- survey depth is isotropic;
- luminosity-function uncertainty is omitted;
- realistic LVK selection is simplified.

Remove sentences about:

- what a curve “does not determine”;
- what a result “should not be interpreted as”;
- what “does not alter the identifiability question.”

Give the missing physics and move on.

---

# 13. CONCLUSIONS SHOULD SOUND LIKE CONCLUSIONS, NOT RUN SUMMARIES

The current Conclusions are still organized as:

- complete run;
- fixed-depth run;
- density run;
- free-density run;
- observational requirements.

Rewrite them in 3 compact paragraphs.

### Paragraph 1
Main scientific result:
\(H_0\) and \(f_{\rm AGN}\) can be measured simultaneously from two tracer fields; quote five-realization numbers.

### Paragraph 2
Main limitation:
uncertainty in the unseen sparse AGN population couples strongly to \(f_{\rm AGN}\) and, for shallow catalogs, to \(H_0\).

### Paragraph 3
Observational consequence:
dense-tracer abundance is much better constrained internally than sparse-tracer abundance; real applications therefore need a well-characterized AGN selection/abundance model.

Do not recount the order in which the experiments were run.

---

# 14. FIGURES: REMOVE ALL TOP LEGEND BARS

This is mandatory.

The current figures still have legend strips sitting above the panels.

They look like analysis figures.

**NO MAIN FIGURE MAY HAVE A LEGEND BAR CENTERED ABOVE THE PANELS.**

No `fig.legend(... loc='upper center' ...)`.

No horizontal strip of color/style samples over the axes.

No legend consuming a band at the top of the figure.

Prefer NO LEGEND AT ALL.

Use direct labels.

---

# 15. FIGURE 1 — DIRECT LABELS

Panel (a):

Do not put a legend above the map.

Directly label a representative:

- AGN point;
- galaxy-hosted GW marker;
- AGN-hosted GW marker

using short annotations in unobtrusive empty regions.

If direct labeling becomes messy, place a compact legend INSIDE a genuinely empty corner at normal text size.

Never above the panel.

Panel (b):

Directly label the two histograms near their peaks/tails:

- galaxy hosts
- AGN hosts

Label the completeness curve next to the curve.

No legend.

---

# 16. FIGURE 2 — NO LEGEND; SMALL MULTIPLES ARE PREFERRED

The current Figure 2 still has `I II III IV V` in a legend strip above the contours.

Remove it completely.

The cleanest solution is now:

### 2 x 3 small-multiple figure

Five panels:
- I
- II
- III
- IV
- V

Each shows the complete-catalog \(H_0-f_{\rm AGN}\) posterior with identical axis limits.

Sixth panel:
summary of all five marginal medians/intervals or simply a compact ensemble summary.

Every posterior panel uses the SAME contour color.

Do not assign a different color to every realization.

The realization identity is the panel label.

This is much cleaner, removes the legend entirely, and lets the reader visually compare posterior location and shape.

Use:

- one dark contour color;
- 50% = thicker;
- 90% = thinner/lighter;
- black input lines.

No legend is required.

If the six-panel format becomes too small, use a 1x5 row spanning two columns plus the marginal summary below.

But the core rule remains:

**realization identity should come from panel labels, not a legend.**

---

# 17. FIGURE 3 — NO LEGEND NEEDED

Figure 3 is already almost fine.

It does not need a legend.

Use:

- heavy bars = 68%;
- light bars = 90%.

Explain this in the caption.

No in-panel legend.

No extra labels.

---

# 18. FIGURE 4 — DIRECT-LABEL EVERY CURVE

The top legend strip on Figure 4 must disappear.

Directly label the curves:

- `99.7%`
- `81.4%`
- `31.5%`
- `9.5%`

Put the labels next to the rightmost or leftmost portion of each curve where they are separated.

Use leader lines only if necessary.

Because the same four styles appear in both panels, panel (a) can carry the direct labels and panel (b) requires none.

Alternatively, directly label both panels if it remains clean.

There should be NO legend object.

If curves overlap too much for direct labels, use small panel-side labels just outside the right spine aligned with each curve endpoint.

Still no traditional legend.

---

# 19. FIGURE 5 — NO TOP LEGENDS

Panel (a):

Directly write:

`galaxies`

and

`AGN`

next to the corresponding marker/error-bar series.

No legend.

Panel (b):

Do NOT overlay three colored realization contours and then add I/II/III in a top legend.

Prefer three small contour panels:

- I
- II
- III

with identical axes and input lines.

This is much more readable and removes the legend problem entirely.

If that makes Figure 5 too wide, arrange:

- left column: abundance-offset summary;
- right column: three vertically stacked mini-contours I/II/III.

Or use a 2x2 layout.

The realization name belongs IN THE PANEL LABEL.

Do not encode realization identity through a legend.

---

# 20. REDUCE COLOR COUNT

The current paper uses too many categorical colors because every realization and every completeness value gets its own color.

Use fewer colors.

A real paper does not need a rainbow to prove that five independent catalogs exist.

Recommended principle:

- galaxy = one color;
- AGN = one color;
- input/truth = black;
- completeness sequence = one restrained sequential family plus line/marker changes;
- independent realization small multiples = same color in each panel.

This will immediately make the figures look less like exploratory analysis.

---

# 21. NO FIGURE TITLES

Do not put titles above axes like:

- complete catalogs
- realization I
- 9.5% completeness

Use panel labels/annotations.

For realizations:

`(a) I`

or

`I`

in the upper-left corner is sufficient.

For physical condition:

small annotation inside the axes if necessary.

---

# 22. CAPTIONS SHOULD BE SHORT

The current captions are still doing too much explanation.

Aim for ~2–4 sentences.

A caption should define:

- what is plotted;
- credible regions;
- black input lines;
- any unusual encoding.

Do not include the interpretation paragraph.

For example Figure 4 does NOT need:

> “so the figure gives a conditional sensitivity for this simulated catalog.”

Delete.

The Results section interprets the curves.

---

# 23. FIGURES MUST BE REDRAWN, NOT PATCHED

Do not merely move legends around.

Open `scripts/make_figures.py` and redesign layout logic.

Use constrained layout / GridSpec deliberately.

Check figure output at the exact size used by AASTeX.

Avoid:

- `bbox_to_anchor` hacks;
- huge top margins reserved for legends;
- tiny legend fonts;
- arbitrary `subplots_adjust(top=...)` just to accommodate labels.

The absence of top legend bands should allow the axes themselves to be larger.

Use that reclaimed space.

---

# 24. FONT AND LINE REQUIREMENTS

At final printed size:

- axes labels ~9 pt;
- ticks ~8 pt;
- direct labels ~8 pt;
- panel labels ~9–10 pt;
- contour lines >=1.3 pt;
- primary curves >=1.5 pt;
- markers >=5 pt.

Do not shrink labels because a plot is crowded.

Redesign the plot.

---

# 25. FIGURE 2 MUST VISUALLY SHOW REPEATED RECOVERY

Do not call it:

- calibration;
- calibrated;
- validation;
- closure suite.

The figure itself should show it.

Five small-multiple posteriors around the same black injected point are enough.

Text:

> “Across five independent realizations, the mean offsets are…”

No adjective necessary.

---

# 26. THE PAPER SHOULD STOP SAYING “SIMULATED CATALOG” EVERY PARAGRAPH

Once Section 3 has defined the mock, later sections do not need constant reminders that it is simulated.

For example, instead of:

> “For the same simulated catalog used in Figure 3…”

write:

> “At fixed merger catalog…”

or simply omit the qualifier if context is clear.

Instead of:

> “for this simulated catalog”

usually delete.

The entire paper is explicitly a mock analysis.

The reader remembers.

---

# 27. DO A HUMAN-SOUNDING PASS, NOT A SYNONYM PASS

After the structural rewrite, read every paragraph aloud.

If it sounds like:

> “We quantify X under Y and evaluate Z in order to isolate…”

rewrite it.

If it sounds like:

> “For complete catalogs, \(H_0\) and \(f_{\rm AGN}\) are only weakly correlated…”

keep it.

Use ordinary astrophysics verbs:

- infer
- measure
- constrain
- increase
- decrease
- shift
- broaden
- trace
- contain
- correlate
- marginalize
- weight

Avoid abstract AI verbs:

- characterize
- elucidate
- isolate
- quantify, repeated
- explore
- interrogate
- establish
- demonstrate
- assess
- evaluate, repeated

---

# 28. TARGET STYLE

The paper should sound like an author who already understands the result and is simply explaining it.

Not like collaborators walking through a notebook together.

No rhetorical flourishes.

No “important insight.”

No “key finding.”

No “our controlled experiments reveal.”

No “this analysis demonstrates.”

No “in the present calculation.”

No “taken together.”

No “overall.”

No “notably.”

No “crucially.”

No “we emphasize.”

Let equations and numbers carry the argument.

---

# 29. AFTER THE REWRITE, SEARCH FOR THESE PHRASES

Run a literal/regex search over the `.tex` files for:

- `we study`
- `we investigate`
- `we explore`
- `we test`
- `designed to`
- `isolate`
- `conditional`
- `same simulated`
- `simulated catalog considered`
- `present calculation`
- `same calculation`
- `this calculation`
- `the calculation`
- `response to`
- `this result`
- `this measurement`
- `this analysis`
- `this test`
- `these simulations`
- `in these mocks`
- `at this depth`
- `at this event count`
- `calibrat`
- `validat`
- `robust`
- `gate`
- `guard`
- `production`
- `benchmark`
- `framework`
- `pipeline`
- `workflow`
- `headline`

Every surviving occurrence must be read in context and justified.

Most should disappear.

Do NOT mechanically replace them.

Rewrite the surrounding sentence.

---

# 30. REBUILD AND READ THE PDF LIKE A REFEREE

Compile the paper.

Then read ONLY the PDF, not the source.

Ask:

### On prose
Does this feel like a finished paper or like an internal experiment report?

### On figures
Do any plots still have:
- top legends;
- tiny keys;
- rainbow realization encoding;
- dashboard-like layout;
- excessive annotations?

If yes, fix them.

### On Results
Can a reader tell the physical result without knowing the order we ran the analyses?

If not, rewrite.

---

# 31. DESIRED FINAL FEEL

The manuscript should read roughly like this:

> Binary mergers assigned to different host populations sample distinct tracers of large-scale structure. We include the host fraction directly in the dark-siren likelihood and infer it together with \(H_0\). For complete catalogs, five independent realizations recover the injected \(H_0\) and \(f_{\rm AGN}\) with little correlation between the two parameters. Catalog incompleteness has little effect when the unobserved-host contribution is known, but the inferred host fraction depends strongly on the abundance assigned to missing AGN. For sufficiently shallow catalogs this uncertainty also shifts \(H_0\). Marginalizing over the tracer abundances constrains the dense galaxy normalization substantially better than the sparse AGN normalization, leaving the latter strongly correlated with \(f_{\rm AGN}\).

That is a scientific narrative.

Do NOT copy this paragraph verbatim.

Use it as the tonal target.

---

# 32. KEEP THE NUMBERS AND CLAIMS EXACTLY GROUNDED

Do not alter the numerical values merely to make prose cleaner.

Retain:

- five-realization complete-catalog offsets;
- planted `f_AGN = 0.30`;
- posterior correlations;
- completeness values;
- density derivatives;
- factor-of-two H0 shift;
- three-realization free-normalization results.

If a sentence becomes awkward because a result is only one-realization, phrase it narrowly.

Do not compensate by writing lab-note disclaimers.

---

# 33. FINAL REPORT

At the end tell me:

1. which sections were substantially rewritten for voice;
2. how many occurrences of “simulated” were removed;
3. how many occurrences of “calculation” were removed;
4. how many occurrences of “isolate” were removed;
5. whether any `calibrat*`, `validat*`, `robust*`, `gate`, `guard`, `production`, `framework`, `pipeline`, or `workflow` remain in normal scientific prose;
6. Figure 1 final labeling strategy;
7. Figure 2 final layout and confirmation that it has NO top legend;
8. Figure 4 confirmation that completeness curves are directly labeled and there is NO legend;
9. Figure 5 final layout and confirmation that there is NO top legend;
10. final page count;
11. build status;
12. confirmation that no scientific number changed.

Do not answer that the paper was “polished.”

Report the actual structural changes.