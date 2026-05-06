# OpenKB Method Adoption

OpenKB was used as the documentation model rather than invoked as a required build step.

## Adopted Ideas

- Persistent compiled wiki instead of re-deriving context for each query.
- Small concept pages with explicit cross-links.
- Source inventory that separates local artifacts, dependencies, and external references.
- Health/verification page for stale or contradictory context.
- Machine-readable manifest for agent routing.
- Knowledge graph file for relationship-level navigation.

## Why The CLI Was Not Required Here

The current task was to restructure and generate markdown documentation inside an existing repo wiki folder. Running the OpenKB CLI would require model/provider configuration and could introduce extra generated state unrelated to the app. The wiki structure remains compatible with OpenKB-style operation because it is plain markdown plus manifest/graph metadata.

## Future OpenKB Integration

If OpenKB is added later:

1. Put raw source documents in a dedicated `wiki/raw/` or external knowledge-base folder.
2. Keep app-generated verification builds out of the OpenKB source set.
3. Point OpenKB output to a separate staging folder.
4. Review diffs before copying approved pages into this wiki.
5. Preserve this wiki's intent-based chunks so implementation agents do not need to ingest full summaries.

