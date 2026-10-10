# Repository guidance

- This is an educational GPT-2 implementation in PyTorch. Keep each change small, readable, and tied to a learning goal.
- Add focused tests when implementing model behavior, and document important tensor shapes and assumptions.
- Keep local datasets, generated checkpoints, and virtual environments out of version control.
- If material is copied from another project, record its source and retain any required copyright and license notices.

## Collaboration

- The owner writes the implementation code in order to learn. Agents explain concepts, provide skeletons (signatures, docstrings with tensor shapes, TODO hints) and tests, then review the owner's code. Write implementations only when asked.
- Before handing over tests, verify them against a private reference implementation kept out of the repository.
- Correctness matters more than training scale: prove the model by matching official GPT-2 weights, and keep training runs small and device-agnostic (`cuda` / `mps` / `cpu`).

## Logs and notes

- `docs/dev-log.md` is the agents' decision log. Append a dated entry whenever a decision is made or changed: what was decided, why, rejected alternatives, and open follow-ups. Also keep its current-status section up to date.
- `notes/` holds the owner's learning notes, later published to the `/blog` section of their personal website (https://xiaonan.dev, repo `Sumire-no-kai/Personal_Website`) and possibly a WeChat official account. The website's current GPT-2 journal content is a draft that these notes will replace, so treat `notes/` as the source and do not align it to the site's draft. The owner writes the content; when asked, agents polish it by fixing technical errors and improving clarity and structure while keeping the owner's voice and viewpoint. Flag doubtful technical claims instead of silently changing their meaning, and do not write whole articles unless asked. Follow the conventions in `notes/README.md`.
- Maintain stage notes under `notes/qa/` so that a reader with no background can follow them: explain concepts, walk through the final code line by line, cross-link related sections, and keep a glossary; include substantive project questions. Integrate explanations beside the functions or tasks they clarify; place questions that merit separate discussion under the related task's "相关问答" subsection, or in the final "延伸问答" section when they concern the whole stage or go beyond it. Use the question itself as the heading (no dates, no separate summary list), and do not transcribe every turn. Preserve the owner's meaning, distinguish stage-wide goals from individual tasks, and keep the main blog articles in the owner's voice.
