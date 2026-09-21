import json
from importlib.resources import files

DEFAULT_ONTOLOGY = 'lynceus-broad-v1'


def load_policy():
    return json.loads(files('lynceus').joinpath('data/policy.json').read_text())


def available_ontologies():
    """Every ontology shipped in the bundle, keyed by its declared id.

    An annotation names the ontology it used. Validation resolves that name here, so a result can
    only claim an ontology this build actually contains.
    """
    registry = {}
    for entry in files('lynceus').joinpath('data').iterdir():
        name = entry.name
        if name.startswith('ontology') and name.endswith('.json'):
            definition = json.loads(entry.read_text())
            registry[definition['id']] = definition
    return registry


def load_ontology(ontology_id=None):
    registry = available_ontologies()
    wanted = ontology_id or DEFAULT_ONTOLOGY
    if wanted not in registry:
        raise ValueError(f'unknown_ontology: {wanted}; available {sorted(registry)}')
    return registry[wanted]


def ancestry(ontology, class_id):
    """Chain from `class_id` up to the root, inclusive, guarding against a cyclic parent chain."""
    index = {c['id']: c for c in ontology['classes']}
    chain, seen = [], set()
    current = class_id
    while current is not None and current not in seen:
        seen.add(current)
        chain.append(current)
        entry = index.get(current)
        if entry is None:
            raise ValueError(f'unknown_class: {current}')
        current = entry['parent']
    return chain


def common_ancestor(ontology, first, second):
    """Lowest class that is an ancestor of both, or None when the ontology has no root in common."""
    up = ancestry(ontology, first)
    other = set(ancestry(ontology, second))
    return next((c for c in up if c in other), None)


def co_occurring(ontology, first, second):
    """Can these two classes both be true of one region?

    Two classes scoring highly on one box are not always competing to describe it. A person on a
    motorcycle is two entities the policy counts separately, and a box around the rider contains
    both, so a detector reporting `person` and `motorcycle` there is describing what is present
    rather than hesitating between readings.

    Only an explicit declaration counts. Inferring co-occurrence from overlap would be the error
    the policy forbids elsewhere, where containment is taken to imply identity.
    """
    declared = ontology.get('co_occurring') or []
    pair = {first, second}
    return any(pair == set(entry) for entry in declared)


def is_useful(ontology, class_id):
    return next((c['useful'] for c in ontology['classes'] if c['id'] == class_id), False)
