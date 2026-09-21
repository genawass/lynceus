from lynceus.policy import load_policy,load_ontology

def test_policy():
    p=load_policy(); assert p['rules']['removable_clothing']=='instance'; assert p['rules']['intrinsic_components']=='part'; assert p['rules']['reflections']=='reflected'
    o=load_ontology();ids={c['id'] for c in o['classes']};assert all(c['parent'] in ids or c['parent'] is None for c in o['classes'])


def test_ontology_registry():
    from lynceus.policy import available_ontologies, load_ontology, DEFAULT_ONTOLOGY
    registry = available_ontologies()
    assert DEFAULT_ONTOLOGY in registry and 'aerial-traffic-v1' in registry
    for name, definition in registry.items():
        ids = {c['id'] for c in definition['classes']}
        assert definition['id'] == name
        assert all(c['parent'] in ids or c['parent'] is None for c in definition['classes'])
        # every ontology reserves the reasons an object can exist without a useful class
        assert {'entity', 'unknown_object'} <= ids
        assert not next(c for c in definition['classes'] if c['id'] == 'unknown_object')['useful']
    import pytest
    with pytest.raises(ValueError, match='unknown_ontology'):
        load_ontology('no-such-ontology')

def test_aerial_ontology_prompts_are_fewer_and_cover_the_domain():
    from lynceus.policy import load_ontology
    aerial = {c['id'] for c in load_ontology('aerial-traffic-v1')['classes'] if c['useful']}
    broad = {c['id'] for c in load_ontology('lynceus-broad-v1')['classes'] if c['useful']}
    assert len(aerial) < len(broad)
    # classes the broad ontology cannot express, which forced motorcycles to be called bicycles
    assert {'motorcycle', 'truck', 'bus', 'van', 'tricycle'} <= aerial
    assert {'motorcycle', 'truck', 'bus', 'van', 'tricycle'} & broad == set()
