# Hints and discoveries: plan (2026-11-06)

Status: plan only, for the user's approval. Nothing is built.

The user asked for useful hints along the way, gamified, that leave room for discovering things on one's own. The app should still stay quiet and not get in the way of the IFC workflow (the user's earlier request).

## Two kinds

- **Hints** say where to look next, once, at the moment they apply. Most are phrased as a question, so the user still finds the answer.
- **Discoveries** are things a user can find out about their building. Before they are found, they show only as a teaser question. When the app sees the user has found one (from the app's state, not a click on "done"), it is named and explained. The discoveries list shows progress, for example "4 of 12 found". This is the gamified part.

## Rules

1. **Placement:**
   - one hint at a time, as a small card at the foot of the current step's panel;
   - never a modal or a toast, and never over the 3D view.
2. **Persistence:** a dismissed hint never returns. Hints already used, for example the code check once run, are skipped. Both are kept in this browser's localStorage.
3. **Switches and reset:** "Hints" can be turned off in one place, and "Show hints again" resets them. Discoveries stay counted.
4. **Accessibility:** cards are announced politely (aria-live), are reachable by keyboard, and do not animate under reduced motion.
5. **Trigger:** a hint appears only when its condition holds for the model on screen, and only after the user has paused for a moment: not during a load, a graph build or a fire run.
6. **Wording:** no claims beyond what the app computed. Fire results stay "uncalibrated"; code values stay "to be confirmed against SBC 201".

## Hints, by step

| Step | When it appears | Hint (draft wording) | Points at |
|---|---|---|---|
| Model | A model has opened and is drawn solid | Can't see inside? Show the building as Ghost. | appearance switch |
| Model | The file marks no exit door | No door here is marked as an exit. Which door would you leave by? You can set the exit yourself in Route. | exit flags, exit picker |
| Model | Generic elements (proxies) are present | Some elements have no type. "What the elements are" can tell what most of them are. | identification |
| Graph | The graph is in more than one part | A cut-off part is often a room whose door is missing from the model. Can you find it? Try Show, then a storey. | parts warning, storey cut |
| Graph | Lifts found, not used | Lifts are left out in a fire. Turn them on and compare: does the route change? | Use lifts |
| Route | First route found | Is there a second way out? Find alternative routes, and see how much longer it is. | alternatives |
| Route | A route exists and the code check has not been run | Is this walk within the limit for an office? Run the code check. | code check |
| Route | Code check run once | Turn on sprinklers and check again: what changes? | sprinkler switch |
| Route | Code check run | Where is the walk tightest? Show the distance to spare on the graph. | distance to spare |
| Simulate | Graph built, no fire origin | Where do fires usually start? "Likely ignition points" ranks them for this building. | ignition ranking |
| Simulate | First fire run finished | Did your route change while the fire spread? Watch the magenta line, or set a start beside the fire. | rerouting |
| Simulate | A Flamescapes run with doors open | Run it again with the doors closed: how far does the smoke get now? | doors switch |
| Simulate | A Flamescapes run finished | Switch to the smoke view: where is visibility lost first? | heat or smoke view |
| Simulate | A fire has run | Leave now or wait? Plan with the forecast gives each place its latest safe time to leave. | forecast plan, deadline map |
| Simulate | A fire has run, no agent trained | Can a learner find the way out on its own? Train an agent and compare it with the certified earliest route. | shielded agent |
| Any | After a full run, once | Take it further: Export for TopologicPy, or Describe this model. | export, describe |

## Discoveries

The teaser is shown before a discovery is found. The app detects it from its own state.

| # | Teaser | Found when | What the card then says |
|---|---|---|---|
| 1 | Is the shortest way out always the right one? | A fire run reroutes the route away from its first path | The shortest route ran towards the fire; the hazard-aware route took another way. |
| 2 | Does every room have a second way out? | Alternative routes found, with the second at least 20% longer | The second way out is much longer: if the first is blocked, this is what is left. |
| 3 | What does the model leave out? | The graph has more than one part | Part of the floor cannot be reached: usually a door or stair missing from the file, not from the building. |
| 4 | When is it too late to leave? | The deadline map shows a place with no safe departure | Some places have no safe time to leave in this fire: they need another way out or earlier warning. |
| 5 | Do sprinklers change the rules? | The code check run with and without sprinklers | With sprinklers the walking limit rises from 60.96 m to 91.44 m, and exits may sit closer together. |
| 6 | Are the exits far enough apart? | The code check reports a separation result | Two exits close together can be lost to one fire: the rule asks for half the diagonal apart (a third with sprinklers). |
| 7 | What do closed doors do? | Flamescapes runs with doors open and with doors closed | Closed doors hold the smoke back: compare the two runs' visibility at head height. |
| 8 | Where does the smoke go first? | The smoke view used during a Flamescapes run | In this run, visibility at head height fell below the limit first at the places shown. |
| 9 | Can embers start a second fire? | A run with embers lights a spot fire | Embers carried by hot gas lit a fire away from the first. |
| 10 | Can a learner be trusted? | The shielded agent certifies its route | The agent's route is checked against the fire, step by step, and certified. |
| 11 | What starts fires here? | An ignition point is used as the fire origin | Fires start where heat meets fuel: the ranking comes from fire statistics for this kind of building. |
| 12 | Is a lift a way out? | A route compared with lifts on and off | Lifts are normally out of use in a fire unless they are evacuation lifts; with them on, the route changed by N m. |

## Where it lives

- **`src/core/hints.js`:** HINTS and DISCOVERIES as data. Each hint has `{id, step, when(state), text, target}`; each discovery has `{id, teaser, found(state, events), text}`. They are pure functions of a small state summary, so they can be unit tested.
- **App wiring:**
  - one `useHints` hook computes the state summary from what App already holds (model, graph, route, fire run, code check, settings);
  - a few events are recorded as they happen (rerouted, spot fire lit, sprinklers compared);
  - the result is kept in localStorage.
- **UI:**
  - a HintCard at the foot of each step panel;
  - a "Discoveries 4/12" badge in the header that opens a small list (teasers and found cards);
  - a Hints switch beside it.
- **Tests:** unit tests for each hint's and discovery's condition. The browser suite checks that a hint shows, dismisses and does not return, and that one discovery is found during the guided run.

## Decisions for the user

1. The name of the list: "Discoveries", "Field notes", or another.
2. Whether to show the progress count (4 of 12) or only the list.
3. Whether hints are on by default. Proposed: yes, with the switch.
4. Whether the guided runs should end on a discovery card.
5. Which hints and discoveries to keep or add.
