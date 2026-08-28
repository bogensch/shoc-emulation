# Genesis / CASS LES Machine-Learning Project Context

## Purpose of this file

This file summarizes the scientific and technical plan discussed with
ChatGPT so that Codex can assist with development of the
machine-learning workflow without needing the prior conversation.

The project is part of the Genesis effort and is focused on
learning/emulating higher-order turbulent moments used by SHOC in EAMxx.
Phase 1 focuses specifically on vertical-velocity moments:

-   `w'^2` --- vertical velocity variance
-   `w'^3` --- third-order vertical velocity moment / skewness-related
    moment

The eventual goal is to deploy the trained emulator in EAMxx/SHOC, so
predictors used by the machine-learning model should, as much as
possible, be quantities that are available to EAMxx at runtime.

## LES training dataset

The training data will come from ERF large-eddy simulations of the CASS
case at the ARM Southern Great Plains (SGP) site.

Planned ensemble:

-   76 CASS days, with one ERF LES simulation for each day
-   12 hours per simulation
-   Domain: 25.6 km x 25.6 km
-   Horizontal resolution: 50 m
-   Vertical resolution: 50 m
-   LES output/statistics will be coarse-grained horizontally to a 3.2
    km x 3.2 km grid
-   Therefore each 3.2-km coarse cell contains 64 x 64 = 4096 native LES
    columns
-   The 25.6-km domain contains 8 x 8 = 64 coarse-grid columns

The intent is for the 3.2-km coarse grid to represent approximately the
information available to a convection-permitting EAMxx grid column,
while the 50-m LES provides the subgrid turbulence truth.

## Machine-learning approach

The current plan is to use a relatively simple neural network rather
than a Fourier Neural Operator (FNO).

The initial architecture should probably be a small feed-forward neural
network / multilayer perceptron (MLP), perhaps 2--4 hidden layers. A
column-aware network may subsequently be tested if vertical context
substantially improves predictions.

The conceptual mapping is:

    EAMxx-compatible atmospheric state
                  |
                  v
          neural network
                  |
                  v
             w'^2, w'^3

An FNO is not currently planned because FNOs are primarily useful for
learning mappings between spatial fields and spatially coupled/PDE-like
behavior. The present problem is closer to a turbulence
parameterization: given the state of an EAMxx column, predict subgrid
turbulent moments.

Start with the simplest architecture possible and increase complexity
only if validation demonstrates that additional spatial/vertical context
is necessary.

## Candidate neural-network predictors

Predictors should be restricted to quantities that EAMxx can provide
during model integration.

Candidate state variables include:

-   liquid-water potential temperature (`theta_l`)
-   total water (`q_t`), or enough information to calculate it from `qv`
    and condensate
-   horizontal wind `u`
-   horizontal wind `v`
-   pressure
-   density
-   height / vertical coordinate
-   layer thickness
-   TKE, provided it is defined consistently with the quantity that will
    be available from SHOC/EAMxx

Potential surface/context predictors include:

-   surface sensible heat flux
-   surface latent heat flux
-   surface stress and/or friction velocity (`u_*`)
-   possibly surface temperature

Useful derived predictors that can be calculated offline from
model-available quantities include:

-   vertical gradients of thermodynamic variables
-   vertical wind shear
-   Brunt--Vaisala frequency / stability measures
-   Richardson number
-   distance from the surface
-   possibly boundary-layer-relative height such as `z/zi`

Do not automatically include every possible variable. The objective is
to determine the minimum physically meaningful EAMxx-compatible
predictor set.

### Resolved vertical velocity

Coarse-grid `w` may be saved during development, but it should not
automatically be used as an NN predictor. Because the target quantities
are subgrid vertical-velocity moments, using resolved `w` could make the
learned relationship overly dependent on resolved dynamics and grid
spacing. This should be tested rather than assumed.

### TKE caveat

Care is required in defining TKE.

The SGS TKE associated with a 50-m LES closure is not necessarily the
TKE that SHOC would know at a 3.2-km coarse-grid scale. Before
production training, define a physically consistent coarse-grid TKE from
the LES velocity fluctuations, potentially including the native LES SGS
contribution as appropriate.

This definition should correspond as closely as possible to the TKE
available to the eventual EAMxx/SHOC emulator.

## LES truth / target calculation

For each 3.2-km coarse cell and vertical level, calculate horizontal
coarse-cell means and turbulent moments from the 50-m LES samples.

For vertical velocity:

    w_bar = mean(w)

    w_prime = w - w_bar

Primary targets:

    w2 = mean(w_prime**2)

    w3 = mean(w_prime**3)

The averaging definition must be documented carefully and used
consistently throughout preprocessing and training.

## Additional LES statistics worth retaining

Even though Phase 1 focuses on `w'^2` and `w'^3`, calculate and retain
additional coarse-grained turbulence statistics when inexpensive to do
so. These may be useful for later Genesis phases and cost very little
compared with storing full LES fields.

Candidates include:

-   `w'theta_l'`
-   `w'q_t'`
-   `theta_l'^2`
-   `q_t'^2`
-   `theta_l' q_t'`

Also retain the coarse means needed to interpret/reconstruct these
statistics.

The philosophy is to avoid having to rerun or reread enormous
native-resolution LES datasets later merely because an inexpensive
coarse-grained turbulence diagnostic was omitted.

## Output frequency

The current recommended starting point is 5-minute coarse-grained
diagnostic output.

For a 12-hour simulation:

    12 hr * 60 min/hr / 5 min = 144 output times

For 76 simulations:

    144 * 76 = 10,944 output times

With 64 coarse columns per time:

    10,944 * 64 = 700,416 coarse-column/time samples

These samples are correlated in space and time and therefore are not
700,416 independent realizations, but the dataset should nevertheless be
large for a relatively simple neural network. The 76 different CASS days
provide particularly important diversity.

Before committing to production output frequency, use one complete LES
day to compare approximately 1-, 5-, and 10-minute sampling and examine
the temporal variability/autocorrelation of `w'^2` and `w'^3`. Five
minutes is expected to be a good compromise. Avoid writing native 50-m
3-D fields every few minutes unless necessary.

Ideally, calculate 3.2-km coarse-grid means and moments during
preprocessing or in ERF diagnostics and write the compact coarse dataset
frequently. Full native-resolution LES snapshots can be written much
less frequently (for example every 30--60 minutes) for debugging and
scientific validation.

## Development strategy

Do NOT wait for all 76 LES simulations before beginning ML development.

Use the first completed 12-hour LES member as a development dataset.

The first-day workflow should establish:

1.  Reading ERF output
2.  Coarse-graining from 50 m to 3.2 km
3.  Calculating coarse means
4.  Calculating `w'^2` and `w'^3`
5.  Calculating optional additional turbulence statistics
6.  Constructing EAMxx-compatible predictors
7.  Defining the coarse-grid TKE correctly
8.  Writing a compact ML-ready dataset
9.  Normalizing/scaling predictors and targets
10. Building a simple PyTorch neural network
11. Training the network
12. Evaluating predictions
13. Plotting predicted versus LES truth
14. Evaluating vertical profiles and time evolution
15. Saving trained weights and all normalization information needed for
    later inference

The one-day model is for software development and debugging, not for
claims about generalization.

## Training / validation / test split

When all 76 days are available, DO NOT randomly split individual
coarse-grid samples into training, validation, and test sets.

Split by entire CASS days.

A possible initial split is approximately:

-   60 days: training
-   8 days: validation
-   8 days: completely held-out testing

The exact split can be adjusted.

This is important because adjacent times and neighboring 3.2-km coarse
cells from the same LES simulation are strongly correlated. Random
sample-level splitting would leak information from a given
meteorological day into both training and testing and would produce an
unrealistically optimistic estimate of generalization.

Consider choosing held-out days so that they sample a useful range of
CASS boundary-layer conditions rather than relying only on a random
split.

## Initial scientific questions for the ML experiments

The development should progressively answer questions such as:

1.  How well can a small MLP reproduce LES `w'^2`?
2.  How well can it reproduce LES `w'^3`?
3.  Is `w'^3` substantially harder to predict than `w'^2`?
4.  Which predictors provide meaningful skill?
5.  How important is TKE?
6.  Does adding vertical context improve prediction?
7.  Can the model generalize to completely unseen CASS days?
8.  Does performance vary systematically with height or boundary-layer
    regime?
9.  Does the emulator behave physically near the surface, inversion, and
    cloud layer?
10. Are predicted `w'^2` values guaranteed or constrained to be
    nonnegative?
11. Are extreme `w'^3` values handled adequately?
12. Can the architecture eventually be implemented cheaply and robustly
    in EAMxx?

## Recommended philosophy for model complexity

Build a baseline before introducing sophisticated ML.

A suggested progression is:

    linear/simple statistical baseline
        ->
    small pointwise MLP
        ->
    MLP with improved/derived physical predictors
        ->
    network with vertical-column context
        ->
    more sophisticated architectures only if justified

Do not move to an FNO simply because it is more sophisticated. An FNO
would become attractive if the scientific problem changes to one where
spatial fields and nonlocal horizontal organization are essential
inputs/outputs.

## Implementation language

Use Python for ML development, with PyTorch as the likely neural-network
framework.

Code should be modular enough that preprocessing, training, evaluation,
and plotting can be run independently. Configuration (variables, paths,
architecture, train/validation/test days, normalization choices, etc.)
should not be hard-coded throughout the source.

Because eventual EAMxx deployment is a goal, keep inference architecture
simple and keep careful records of:

-   exact input variables
-   ordering of inputs
-   units
-   transformations
-   normalization constants
-   network architecture
-   activation functions
-   trained weights
-   target transformations
-   output units

Reproducibility is important.

## Immediate next step when the first ERF member is available

Before writing the final training system, inspect the first ERF dataset
and determine:

-   exact ERF variable names
-   array dimensions and staggering
-   vertical coordinate
-   output cadence
-   units
-   how the 25.6-km domain maps exactly into the 8 x 8 coarse grid
-   how coarse averaging should handle any staggered velocity fields
-   exact definitions of thermodynamic predictors
-   exact definition of coarse-grid TKE
-   exact definitions of `w'^2` and `w'^3`
-   desired ML-ready file format and dimensions

Then implement and validate the preprocessing pipeline before scaling it
to all 76 days.

## Key project principle

The final emulator should learn a relationship that EAMxx can actually
use:

    information available to EAMxx/SHOC
                  ->
       LES-informed SGS moments

Do not allow the training system to obtain skill from information that
will not exist when the neural network is eventually called inside
EAMxx.
