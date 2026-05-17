MATCH p=(team:Page)-[rels*1..2]-(candidate:Page)
WHERE team.title IN ['Sub', 'Sandwich'] AND NOT candidate.title IN ['Sub', 'Sandwich']
WITH candidate, collect(DISTINCT team.title) as targets, min(size(rels)) as dist_to_team
WHERE size(targets) >= 1
RETURN candidate.title, targets, dist_to_team
LIMIT 5;
