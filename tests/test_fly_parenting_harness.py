import json
from information_boltzmann.core.fly_parenting_harness import ParentingHarness, conversation_health
from information_boltzmann.core.fly_mother_world import world_prompt, activity_prompt, GeminiApiMother


def setup(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'parenting': {'education_stage': 'connection', 'activity': {'kind': 'shared_reading'}},
                                'harness': {'check_every_new_mother_turns': 12}}), encoding='utf8')
    return plan, ParentingHarness(plan, tmp_path / 'state.json', tmp_path / 'notes.md')


def monitor(turns=12):
    return {'mother_turns': turns, 'recent_fly_turns': 50, 'fly_spoken_tokens': 60,
            'mother_word_count': 200, 'fly_repeat_bigrams': .8, 'mother_repeat_bigrams': .5}


def test_same_snapshot_cannot_mature_or_trigger_three_observations(tmp_path):
    plan, h = setup(tmp_path)
    before = plan.read_bytes()
    for _ in range(6):
        result = h.update(trained=100, monitor=monitor())
    assert result['action'] == 'already_observed'
    assert len(h.history) == 1
    assert before == plan.read_bytes()


def test_new_windows_without_new_interactions_do_not_count(tmp_path):
    _, h = setup(tmp_path)
    h.update(trained=100, monitor=monitor())
    result = h.update(trained=108, monitor=monitor(13))
    assert result['action'] == 'collect_more_interaction'
    assert len(h.history) == 1


def test_ling_remains_caregiver_and_nll_does_not_promote(tmp_path):
    plan, h = setup(tmp_path)
    before = plan.read_bytes()
    for i in range(3):
        out = h.update(trained=100+i*100, monitor=monitor(12+i*12), reply_nll=2, reply_unigram=8)
    assert out['action'] == 'caregiver_review_echo'
    assert not out['changed']
    assert plan.read_bytes() == before
    assert all(x['status'] == 'unmeasured' for x in out['brain_status'].values())
    out = h.update(trained=500, monitor=monitor(60), apply=True)
    assert out['changed']
    saved = json.loads(plan.read_text(encoding='utf8'))
    assert saved['parenting']['education_stage'] == 'connection'
    assert saved['parenting']['activity']['kind'] == 'shared_reading'


def test_silence_not_counted_as_repeated_speech_or_cross_turn_bigrams():
    m = conversation_health([('fly','...'), ('fly','...'), ('fly','boat'), ('fly','boat'),
                             ('mother','Mengli, look.'), ('mother','Look, Mengli.')])
    assert m['fly_identical_consecutive_turns'] == 1
    assert m['fly_repeat_bigrams'] == 0
    assert m['fly_silent_turns'] == .5
    assert m['mother_name_prefix_share'] == .5


def test_memory_retrieval_does_not_prime_unrelated_cat(tmp_path):
    memory = tmp_path / 'world.json'
    memory.write_text(json.dumps({'facts':[{'fact':'Katz sleeps on the chair'},
                                          {'fact':'Papa plays the drum'}]}),encoding='utf8')
    assert world_prompt(memory, query='paper boat water') == ''
    result = world_prompt(memory, query='drum')
    assert 'Papa' in result and 'Katz' not in result


def test_shared_reading_exposes_current_page_only_and_keeps_names():
    activity = {'page':0, 'pages':['A paper boat floats.', 'A secret next event.']}
    m = GeminiApiMother.__new__(GeminiApiMother)
    m._refresh_plan = lambda: None
    m.base_system = 'A caring mother.'
    m.plan = {'parenting': {'activity':activity, 'current_topic':'paper boat'},
              'family':{'child':{'name':'Mengli'},'mother':{'name':'Mama'},'father':{'name':'Papa'}}}
    stable, volatile = m.get_prompt_parts([('fly','boat')])
    assert 'Mengli' in stable and 'Mama' in stable and 'Papa' in stable
    assert 'A paper boat floats.' in volatile
    assert 'secret next event' not in volatile
    assert 'Katz' not in volatile


def test_free_play_instructions_reach_mother_without_story_pages():
    result = activity_prompt({'parenting': {'activity': {
        'kind': 'free_play', 'instruction': 'Take turns rolling the ball.'}}})
    assert 'Take turns rolling the ball.' in result


def test_interleaved_resources_are_available_together_and_not_time_gated(tmp_path):
    from information_boltzmann.core.fly_parenting_harness import learning_resources_prompt
    shelf = tmp_path / 'shelf.json'
    shelf.write_text(json.dumps({'entries': [
        {'id': 'story', 'title': 'Water', 'strand': 'causal', 'source': 21, 'text': 'A crow raised the water.', 'use': 'Read together.'},
        {'id': 'rhyme', 'title': 'Count', 'strand': 'quantity', 'source': 24623, 'text': 'One two three.', 'use': 'Enjoy the rhyme.'}]}), encoding='utf8')
    plan = {'parenting': {'education_stage': 'connection', 'interleaved_learning': {
        'enabled': True, 'shelf_path': str(shelf), 'strands': [
            {'invitation': 'relationships and pronouns'}, {'invitation': 'quantities in everyday events'}]}}}
    result = learning_resources_prompt(plan, [('fly', 'water')])
    assert 'relationships and pronouns' in result and 'quantities in everyday events' in result
    assert 'A crow raised the water.' in result and 'One two three.' in result
    assert 'No timed subject blocks' in result
    assert 'not a subject gate' in result


def test_word_entropy_reports_concentration_with_sample_counts():
    low = conversation_health([('mother', 'boat boat boat boat')])
    high = conversation_health([('mother', 'boat leaf rain sun')])
    assert low['mother_word_entropy_bits'] == 0
    assert high['mother_word_entropy_bits'] == 2
    assert high['mother_effective_vocabulary'] == 4
    assert high['mother_word_count'] == low['mother_word_count'] == 4


def test_interest_based_shelf_does_not_turn_function_words_into_story_requests(tmp_path):
    from information_boltzmann.core.fly_parenting_harness import learning_resources_prompt
    shelf = tmp_path / 'shelf.json'
    shelf.write_text(json.dumps({'sources': [{'title': 'Stories'}, {'title': 'Rhymes'}], 'entries': [
        {'id': 'story', 'title': 'Water', 'strand': 'causal', 'source': 21,
         'text': "She went after the crow to find water. We're here; she'll bring tea for M.", 'use': 'Read together.'},
        {'id': 'rhyme', 'title': 'Count', 'strand': 'quantity', 'source': 24623,
         'text': 'One two three.', 'use': 'Enjoy the rhyme.'}]}), encoding='utf8')
    plan = {'parenting': {'interleaved_learning': {
        'enabled': True, 'excerpt_policy': 'on_interest', 'shelf_path': str(shelf),
        'strands': [{'invitation': 'relationships'}, {'invitation': 'quantity'}]}}}
    for fragment in ('I', 'She the', 'when', 'it perhaps goes', "I're", "'ll OF,", 'M'):
        result = learning_resources_prompt(plan, [('mother', 'She went after the crow.'), ('fly', fragment)])
        assert 'relationships' in result and 'quantity' in result and 'Available books:' in result
        assert 'She went after the crow to find water.' not in result
        assert 'One two three.' not in result
    relevant = learning_resources_prompt(plan, [('fly', 'water')])
    assert 'She went after the crow to find water.' in relevant
    assert 'One two three.' not in relevant
    requested = learning_resources_prompt(plan, [('fly', 'read a story')])
    assert 'She went after the crow to find water.' in requested
    assert 'One two three.' in requested


def test_home_memory_does_not_retrieve_cat_from_she_or_we(tmp_path):
    memory = tmp_path / 'world.json'
    memory.write_text(json.dumps({'facts': [
        'We have a cat called Katz; she sleeps on a chair.',
        'A red ball rolls when touched.']}), encoding='utf8')
    for fragment in ('She the', 'despite We I', 'when'):
        assert world_prompt(memory, query=fragment) == ''
    assert 'Katz' in world_prompt(memory, query='cat')
    assert 'red ball' in world_prompt(memory, query='ball')


def test_nonverbal_event_does_not_retrieve_quiet_clock(tmp_path):
    memory = tmp_path / 'world.json'
    memory.write_text(json.dumps({'facts': [
        'The clock has a quiet steady rhythm in the hallway.']}), encoding='utf8')
    assert world_prompt(memory, query='<quiet_turn/> stories and rhymes') == ''
    assert 'clock' in world_prompt(memory, query='quiet')


def test_deepseek_request_separates_caregiver_notes_and_literal_child_speech():
    from information_boltzmann.core.fly_mother_world import DeepSeekMother
    from types import SimpleNamespace
    m = DeepSeekMother.__new__(DeepSeekMother)
    m._refresh_plan = lambda: None
    m.base_system = 'A caring mother.'
    m.plan = {'family': {'child': {'name': 'Lina'}, 'mother': {'full': 'Mama Yun', 'name': 'Mama'}},
              'parenting': {'intention': 'Follow the child.', 'context_correction': 'Lina is the child.'}}
    m.model, m.temperature, m.max_tokens, m.retries = 'fixture', .7, 60, 2
    m.api_key, m.url, m.thinking_field = 'fixture', 'fixture', False
    captured = []
    def post(url, *, json, **kwargs):
        import copy
        captured.append(copy.deepcopy(json))
        return SimpleNamespace(status_code=200, json=lambda: {
            'choices': [{'message': {'content': 'You said your name!' if len(captured) == 1 else 'I hear you.'}}]})
    m.requests = SimpleNamespace(post=post)
    history = [('mother', 'Hello, Lina.'), ('fly', 'Li')]
    assert m.reply(history) == 'I hear you.'
    messages = captured[0]['messages']
    assert len(captured) == 2
    assert m.last_grounding_rejections[0]['flags'] == ['unsupported_complete_child_name_claim']
    assert messages[1] == {'role': 'assistant', 'content': 'Hello, Lina.'}
    assert any(msg == {'role': 'user', 'content': 'Li'} for msg in messages)
    assert 'Assistant / adult speaker: Mama Yun' in messages[0]['content']
    assert 'User / child speaker: Lina' in messages[0]['content']
    # Volatile caregiver notes ride at the tail to keep messages[0] prefix cache-friendly
    assert any('Follow the child.' in msg['content'] for msg in messages[1:])
    assert any('Caregiver correction' in msg['content'] for msg in messages[1:])
    # Grounding check appends retry guidance at the tail
    assert any('Caregiver grounding check' in msg['content'] for msg in captured[1]['messages'])
    assert history[-1] == ('fly', 'Li')


def test_caregiver_identity_and_misquote_flags_do_not_rewrite_names():
    from information_boltzmann.core.fly_parenting_harness import caregiver_grounding_review
    transcript = [('fly', 'Meng'), ('mother', 'You said my name — Mengli!'),
                  ('fly', 'Meng'), ('mother', 'You said "Mengli".'),
                  ('fly', 'Mama'), ('mother', 'You said my name, Mama. You said "Mama".')]
    before = list(transcript)
    result = caregiver_grounding_review(transcript)
    assert [f['kind'] for f in result['flags']] == [
        'possible_speaker_identity_confusion', 'unsupported_adult_name_claim', 'possible_inaccurate_child_quote']
    assert transcript == before and not result['automatic_rewrite']


def test_name_grounding_allows_purposeful_calls_and_actual_name_speech():
    from information_boltzmann.core.fly_parenting_harness import caregiver_reply_flags
    assert not caregiver_reply_flags('Meng', 'Mengli, I hear you.')
    assert not caregiver_reply_flags('Mengli', 'You said your name, Mengli.')
    assert not caregiver_reply_flags('Mama', 'You said my name, Mama.')
    assert caregiver_reply_flags('Meng', 'You said your own name!') == ['unsupported_complete_child_name_claim']


def test_stage_readiness_needs_reviewed_cross_session_context_evidence():
    from information_boltzmann.core.fly_parenting_harness import stage_readiness
    criterion = {'id':'recognition','status':'passed','evidence':[]}
    curriculum = {'stages':[{'id':'connection','criteria':[criterion]}]}
    assert not stage_readiness(curriculum, 'connection')['ready']
    criterion['evidence'] = [{'reviewer':'Ling','session':str(i),'context':'same','quote':'observed reply','result':'pass'} for i in range(3)]
    assert not stage_readiness(curriculum, 'connection')['ready']
    criterion['evidence'][2]['context'] = 'changed context'
    assert stage_readiness(curriculum, 'connection')['ready']


def test_ling_audit_uses_local_run_and_token_weighted_latest_scores(tmp_path, monkeypatch):
    from scripts.ib.fly_interface import ling_harness
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ling_harness, 'ROOT', tmp_path)
    harness = tmp_path / 'data/harness'
    harness.mkdir(parents=True)
    (harness/'raising.json').write_text(json.dumps({'parenting':{'education_stage':'connection'}}),encoding='utf8')
    # A different individual's global snapshot must never enter this review.
    (harness/'monitor.json').write_text(json.dumps({'monitor':{'fly_ttr':999}}),encoding='utf8')
    run = tmp_path/'child'; run.mkdir()
    (run/'mother.jsonl').write_text(json.dumps({'who':'fly','text':'boat boat'})+'\n',encoding='utf8')
    for chunk, n, ce in ((2,9,3.),(10,1,9.)):
        (run/f'scores_c{chunk}.jsonl').write_text(json.dumps({'trained':chunk*100,'scores':[ce]*n,'unigram':[10.]*n,'sources':['mother_reply']*n})+'\n',encoding='utf8')
    report = ling_harness.audit(run, tmp_path/'checkpoint')
    assert report['world']['fly_ttr'] == .5
    assert report['normal_reply']['tokens'] == 10
    assert abs(report['normal_reply']['mean_gain_over_unigram'] - 6.4) < 1e-9
    assert not report['stage_readiness']['ready']
    assert 'brain_healthy' not in report['brain']
    assert report['brain']['checks']['read_think_speak']['status'] == 'needs_current_paired_evidence'
