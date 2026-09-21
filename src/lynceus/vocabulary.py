"""Per-model prompt vocabularies, versioned apart from the ontology.

A concept-promptable model is driven by noun phrases and its behaviour changes with the phrasing,
so an ontology ID -- an identifier chosen for a schema -- is not a prompt. The phrasing is also not
shared between models: a CLIP-style text tower, a presence head and a phrase grounder do not
respond alike, and wording that finds a small overhead figure for one may find nothing for another.

So a vocabulary is its own artifact, one per model family, mapping the classes of a named ontology
to the phrasing that model is asked with. Keeping it out of the ontology is what stops a prompt
recalibration from bumping the taxonomy's version and invalidating qualification results that have
nothing to do with taxonomy.

A class the vocabulary does not cover falls back to the ontology's canonical name, and the fallback
is recorded rather than silently applied, so an uncalibrated class is visible in the run.

Selecting a vocabulary is dataset-level prompt optimization. It is fitted on a development or
calibration split and frozen before the locked evaluation is opened; one tuned on the evaluation
panel invalidates that panel.
"""
import json
from pathlib import Path

from .artifacts import canonical, digest
from .policy import available_ontologies

DATA=Path(__file__).parent/'data'
REQUIRED=('id','version','adapter','ontology','prompts')


def available_vocabularies():
    found={}
    for path in sorted(DATA.glob('vocabulary-*.json')):
        spec=json.loads(path.read_text())
        found[spec['id']]=spec
    return found


def load_vocabulary(identifier):
    if identifier is None:return None
    registry=available_vocabularies()
    if identifier not in registry:raise ValueError('unknown_vocabulary: '+str(identifier))
    return registry[identifier]


def validate_vocabulary(spec):
    """A vocabulary must name what it maps from and what it is for, and cover only real classes."""
    missing=[k for k in REQUIRED if k not in spec]
    if missing:raise ValueError('missing_vocabulary_field: '+','.join(missing))
    registry=available_ontologies()
    if spec['ontology'] not in registry:raise ValueError('unknown_ontology: '+str(spec['ontology']))
    ontology=registry[spec['ontology']]
    known={c['id'] for c in ontology['classes']}
    unknown=sorted(set(spec['prompts'])-known)
    if unknown:raise ValueError('vocabulary_covers_unknown_class: '+','.join(unknown[:3]))
    for class_id,phrase in spec['prompts'].items():
        if not isinstance(phrase,str) or not phrase.strip():
            raise ValueError('empty_prompt_for_class: '+class_id)
    return spec


def resolve_prompts(ontology,vocabulary=None,classes=None):
    """The phrasing to query each class with, and where each one came from.

    Returns the class ids in a fixed order, the prompt for each, and the list of classes that fell
    back to their canonical name. The fallback list is the point: a vocabulary that covers half the
    ontology should not look like one that covers all of it.
    """
    selected=[c for c in ontology['classes'] if (c['id'] in classes if classes is not None else c['useful'])]
    prompts=(vocabulary or {}).get('prompts',{})
    resolved=[];fallbacks=[]
    for entry in selected:
        phrase=prompts.get(entry['id'])
        if phrase is None:
            phrase=entry['name'];fallbacks.append(entry['id'])
        resolved.append({'id':entry['id'],'prompt':phrase})
    return {'classes':[r['id'] for r in resolved],'prompts':[r['prompt'] for r in resolved],
            'fallbacks':fallbacks,
            'vocabulary':None if vocabulary is None else {
                'id':vocabulary['id'],'version':vocabulary['version'],'adapter':vocabulary['adapter'],
                'ontology':vocabulary['ontology'],'sha256':digest(canonical(vocabulary))},
            # An uncalibrated run is a legitimate run; it just must not look like a calibrated one.
            'coverage':round(1-len(fallbacks)/len(resolved),4) if resolved else None}


def check_adapter(resolution,adapter):
    """A vocabulary tuned for one model must not be handed to another.

    The whole reason vocabularies are per-model is that phrasing does not transfer, so using one
    across adapters would reintroduce exactly the error the split exists to prevent.
    """
    declared=(resolution.get('vocabulary') or {}).get('adapter')
    if declared is not None and declared!=adapter:
        raise ValueError(f'vocabulary_adapter_mismatch: built for {declared}, used by {adapter}')
    return resolution
