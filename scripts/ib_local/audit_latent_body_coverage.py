"""Build a conservative, source-linked body audit against the proven index.

Only declarations explicitly read and connected to executed code are assessed.
All others remain unreviewed, not false. Local numerical instantiation does not
mean Lean verified Python, or that an organism-level contract is satisfied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT/'docs/information_boltzmann/CGB_PROVEN_INDEX.md'
OUTPUT = ROOT/'results/published/cgb_latent_body_coverage_20261104.json'
DECLARATION = re.compile(r'^- \[`([^`]+)`\]\(([^)#]+)#L(\d+)\)', re.M)

# status per body: realized_local means the stated mathematical component is
# present with cited premises; not a complete theorem instantiation for every
# body state, nor a formal refinement proof. Coverage is declaration-counted,
# not a number of theorem-dependent proof steps or witnessed life properties.
AUDIT = {
    'CGB.Generation.tiedWrite_eq_delta_add': dict(
        old='realized_local', new='realized_local',
        premise='one-hot keyed value; frozen current row; beta as declared',
        old_code='scripts/ib_local/ngram_fast_table.py:_run row update',
        new_code='scripts/ib_local/latent_navigation.py:LatentTransitionTable.append_pairs beta=1',
        evidence='tests/test_latent_navigation.py:test_one_hot_write_preserves_other_slots_and_exact_successor'),
    'CGB.Generation.tied_write_is_error_descent': dict(
        old='realized_local', new='realized_local',
        premise='one-hot keyed local squared error; not full language objective',
        old_code='scripts/ib_local/ngram_fast_table.py:_run',
        new_code='scripts/ib_local/latent_navigation.py:LatentTransitionTable.append_pairs beta=1',
        evidence='lean/CGBAxioms/reviews/latent_navigation_admission_20261104.lean'),
    'CGB.Generation.tied_write_error_shrinks': dict(
        old='realized_local', new='realized_local',
        premise='unit one-hot key, local write; new endpoint beta=1, raw stored coordinate',
        old_code='scripts/ib_local/ngram_fast_table.py:_run',
        new_code='scripts/ib_local/latent_navigation.py:LatentTransitionTable.append_pairs',
        evidence='lean/CGBAxioms/reviews/latent_navigation_admission_20261104.lean; tests/test_latent_navigation.py:test_one_hot_write_preserves_other_slots_and_exact_successor'),
    'CGB.Generation.tied_write_cross_read': dict(
        old='realized_local', new='realized_local',
        premise='raw value-row update, fixed linear query and fixed encoder; excludes key/softmax changes',
        old_code='scripts/ib_local/ngram_fast_table.py:_run',
        new_code='scripts/ib_local/latent_navigation.py:LatentTransitionTable.append_pairs',
        evidence='lean/CGBAxioms/reviews/latent_navigation_admission_20261104.lean; tests/test_latent_navigation.py:test_one_hot_write_preserves_other_slots_and_exact_successor'),
    'CGB.Generation.exact_prediction_no_write': dict(
        old='realized_local', new='realized_local',
        premise='only the value coordinate when incoming value equals stored value; replacement can still change the key',
        old_code='scripts/ib_local/ngram_fast_table.py:_run',
        new_code='scripts/ib_local/latent_navigation.py:LatentTransitionTable.append_pairs',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean'),
    'CGB.ClosedLoopCredit.objective_deriv_advantage': dict(
        old='conditional_interface', new='not_connected',
        premise='old action bridge only: complete action record, fixed conditional world kernel and direct-score derivative',
        old_code='scripts/ib_local/language_action.py:joint_loss',
        new_code='scripts/ib_local/latent_navigation.py; not connected to action trainer',
        evidence='tests/test_language_action.py:test_joint_estimator_matches_exact_expected_shared_parameter_derivative'),
    'CGB.ActiveInference.action_condition_required': dict(
        old='conditional_interface', new='not_connected',
        premise='world must respond differently to actions and prediction must use resulting self-conditioned state',
        old_code='scripts/ib_local/language_action.py:TextChoiceWorld',
        new_code='scripts/ib_local/latent_navigation.py:predict_successor, not yet wired to dialogue',
        evidence='tests/test_language_action.py:test_world_intervention_and_cursor_restore_are_real_and_bounded'),
    'CGB.LanguageLifeKernel.delayed_action_credit_requires_state_separation': dict(
        old='conditional_interface', new='not_connected',
        premise='later state must retain distinctions whose actual consequences require distinct credit',
        old_code='scripts/ib_local/language_action.py:self-heard byte',
        new_code='scripts/ib_local/latent_navigation.py:memory, no action-credit settlement',
        evidence='lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean'),
    'CGB.CountedPopulation.tag_is_exact_credit': dict(
        old='not_instantiated', new='not_instantiated',
        premise='tag follows specified nonnegative linear counted-population transport and fixed bets',
        old_code='scripts/ib_local/retrieved_table_credit.py:retrieval derivative, not past-write tag',
        new_code='scripts/ib_local/latent_navigation.py:no historical causal eligibility',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean'),
    'CGB.InformationEnergy.mixture_regret_le': dict(
        old='not_instantiated', new='not_instantiated',
        premise='Bayesian mixture over fixed hypothesis sequence likelihoods, not same-batch learning progress',
        old_code='scripts/ib_local/language_action.py',
        new_code='scripts/ib_local/latent_navigation.py',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean'),
    'CGB.InformationEconomics.causal_carrier_contract_iff': dict(
        old='unverified', new='unverified',
        premise='all required readout and successor distinctions for the declared future family',
        old_code='scripts/ib_local/train_gdn_fast_table.py:TableGDN',
        new_code='scripts/ib_local/latent_navigation.py:LatentNavigationBody',
        evidence='lean/CGBAxioms/CGBAxioms/InformationEconomics.lean'),
    'CGB.Generation.exact_key_capacity': dict(
        old='constraint_only', new='constraint_only',
        premise='linear independent keys in finite d-dimensional vector space; not capacity of all mixed associations',
        old_code='scripts/ib_local/gated_deltanet.py',
        new_code='scripts/ib_local/latent_navigation.py',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean'),
    'CGB.Generation.gated_delta_contracts': dict(
        old='different_recurrence', new='different_recurrence',
        premise='alpha times deltaStep, not assumed for decay_then_delta runtime',
        old_code='scripts/ib_local/gated_deltanet.py:decay_then_delta',
        new_code='scripts/ib_local/gated_deltanet.py:decay_then_delta',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean'),
}

AUDIT.update({
    'CGB.Generation.ActiveNavigation.reactivation_exact_held': dict(
        old='not_instantiated',new='realized_local',v2='realized_local',
        premise='fixed detached source, fixed query weight, a=1; value-path derivative with current encoder only',
        old_code='scripts/ib_local/ngram_fast_table.py:no encoder',
        new_code='scripts/ib_local/latent_navigation.py:LatentTransitionTable.read',
        v2_code='scripts/ib_local/latent_navigation_v2.py:TransitionMemory.read',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean; tests/test_latent_navigation_v2.py:test_retained_raw_values_reencode_with_exact_frozen_weight_local_gradient'),
    'CGB.Generation.ActiveNavigation.reactivation_superposition': dict(
        old='not_instantiated',new='realized_local',v2='realized_local',
        premise='fixed raw source features and fixed mixing weights; excludes weight derivative and original GDN parameter drift',
        old_code='scripts/ib_local/ngram_fast_table.py:no encoder',
        new_code='scripts/ib_local/latent_navigation.py:LatentTransitionTable.read',
        v2_code='scripts/ib_local/latent_navigation_v2.py:TransitionMemory.read',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean; tests/test_latent_navigation_v2.py:test_retained_raw_values_reencode_with_exact_frozen_weight_local_gradient'),
    'CGB.Generation.ActiveNavigation.reactivation_partial_bias': dict(
        old='not_instantiated',new='constraint_only',v2='constraint_only',
        premise='fixed linear superposition; retained and omitted contributions must be explicitly identifiable',
        old_code='scripts/ib_local/ngram_fast_table.py:no encoder',
        new_code='scripts/ib_local/latent_navigation.py:retained feature table',
        v2_code='scripts/ib_local/latent_navigation_v2.py:merge and eviction obscure individual omitted terms',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean'),
    'CGB.ActiveInference.active_closure': dict(
        old='unverified',new='unverified',v2='unverified',
        premise='unbounded usable signal, bounded signal for every eventually passive policy, and action-sensitive response',
        old_code='scripts/ib_local/language_action.py',
        new_code='scripts/ib_local/latent_navigation.py',
        v2_code='scripts/ib_local/language_navigation_v2.py and language_navigation_world.py',
        evidence='lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean'),
})

V2_OVERRIDES = {
    'CGB.ClosedLoopCredit.objective_deriv_advantage': 'partial_estimator',
    'CGB.ActiveInference.action_condition_required': 'conditional_interface',
    'CGB.LanguageLifeKernel.delayed_action_credit_requires_state_separation': 'conditional_interface',
}

# One scoring family realized on the typed reply-acceptance channel. Five
# source declarations describe related local algebra, not five new organs.
TYPED_WOOD_AUDIT = {
    name: dict(status='realized_local',
        premise='Bernoulli outcome law q fixed for the expectation; actual body emits bounded p before observing y; no empirical calibration claim',
        code='scripts/ib_local/language_navigation_kanli_v4t.py; inherited _outcome_loss trains squared error on the typed probability',
        evidence='tests/test_navigation_typed_wood.py; lean/CGBAxioms/reviews/typed_wood_admission_20261105.lean',
        runtime_evidence='results/cgb_navigation_typed_wood_20261105/calibration.json; results/cgb_navigation_typed_wood_20261105/RESULT.md',
        scope='binary scoring channel only; world-specific suffix-match outcome, not general usefulness')
    for name in (
        'CGB.Unified.bernoulli_expected_brier_gap',
        'CGB.Unified.bernoulli_expected_brier_calibrated',
        'CGB.Unified.bernoulli_prediction_error_expectation',
        'CGB.Unified.brier_score_is_not_outcome_prediction_error',
        'CGB.Unified.calibrated_evidence_penalty_is_square')
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    index_text = INDEX.read_text(encoding='utf-8')
    declarations = DECLARATION.findall(index_text)
    names = [name for name, _, _ in declarations]
    if len(names) != len(set(names)) or not (AUDIT.keys() | TYPED_WOOD_AUDIT.keys()) <= set(names):
        raise ValueError('index changed or has duplicate/missing audited declarations')
    rows = []
    for name, relative_source, line in declarations:
        row = dict(name=name, source=str((INDEX.parent/relative_source).resolve().relative_to(ROOT)).replace('\\', '/'),
                   line=int(line))
        row.update(AUDIT.get(name, dict(old='unreviewed', new='unreviewed')))
        if name in AUDIT:
            row.setdefault('v2', V2_OVERRIDES.get(name, row['new']))
            row.setdefault('v2_code', row.get('new_code'))
        else:
            row['v2'] = 'unreviewed'
        typed = TYPED_WOOD_AUDIT.get(name)
        row['v4t'] = typed['status'] if typed else 'unreviewed'
        if typed:
            row['v4t_contract'] = typed
        rows.append(row)
    return dict(
        description='Partial source-and-runtime review. A local instance is not organism-wide theorem coverage.',
        index_sha256=sha(INDEX), inventory_count=len(rows),
        source_hashes={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [
            ROOT/'scripts/ib_local/latent_navigation.py', ROOT/'scripts/ib_local/ngram_fast_table.py',
            ROOT/'scripts/ib_local/language_action.py', ROOT/'tests/test_latent_navigation.py',
            ROOT/'lean/CGBAxioms/reviews/latent_navigation_admission_20261104.lean',
            ROOT/'scripts/ib_local/latent_navigation_v2.py', ROOT/'scripts/ib_local/language_navigation_v2.py',
            ROOT/'scripts/ib_local/language_navigation_world.py',
            ROOT/'tests/test_latent_navigation_v2.py',
            ROOT/'lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean',
            ROOT/'lean/CGBAxioms/reviews/latent_transition_merge_20261104.lean',
            ROOT/'lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean',
            ROOT/'lean/CGBAxioms/reviews/typed_wood_admission_20261105.lean',
            ROOT/'scripts/ib_local/language_navigation_kanli_v4p.py',
            ROOT/'scripts/ib_local/language_navigation_kanli_v4q.py',
            ROOT/'scripts/ib_local/language_navigation_kanli_v4r.py',
            ROOT/'scripts/ib_local/language_navigation_kanli_v4s.py',
            ROOT/'scripts/ib_local/language_navigation_kanli_v4t.py',
            ROOT/'tests/test_navigation_typed_wood.py']},
        status_counts={body: dict(Counter(row[body] for row in rows)) for body in ('old', 'new', 'v2', 'v4t')},
        assessed_count=len(AUDIT.keys() | TYPED_WOOD_AUDIT.keys()),
        assessed_by_body={'old':len(AUDIT),'new':len(AUDIT),'v2':len(AUDIT),'v4t':len(TYPED_WOOD_AUDIT)},
        typed_scoring_family_count=1, declarations=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rendered = json.dumps(build(), ensure_ascii=False, indent=2)+'\n'
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding='utf-8') != rendered:
            raise SystemExit('coverage audit stale: rebuild and check its source hashes')
    else:
        OUTPUT.write_text(rendered, encoding='utf-8')
    result = json.loads(rendered)
    print(json.dumps({k: result[k] for k in ('inventory_count', 'assessed_count', 'status_counts')}, indent=2))


if __name__ == '__main__':
    main()
