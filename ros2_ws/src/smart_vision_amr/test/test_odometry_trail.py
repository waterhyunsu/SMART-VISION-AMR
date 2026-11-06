"""Odometry source continuity is separate from map localization corrections."""
from math import cos, sin
import pytest
from smart_vision_amr.odometry_trail import OdometryTrail


def sample(t,x,y=0.,yaw=0.,frame='odom'):
    return (t,round(t*1e9),x,y,yaw,frame)


def straight():
    trail=OdometryTrail()
    for i in range(13):trail.observe(sample(100.+i*.1,.025*i),'base')
    return trail


def test_actual_trace_reprojects_rigidly_after_amcl_correction():
    trail=straight();actual=list(trail.points);paired=sample(101.2,.3)
    a=trail.project((1.3,2.,0.),paired,now=101.2)
    b=trail.project((-.5,.8,1.),paired,now=101.2)
    assert list(trail.points)==actual
    assert len(a)==len(b)>=10
    for old,new in zip(a,b):
        # Relative vectors rotate as a single rigid body; no fictitious travel.
        dx,dy=old.x-a[-1].x,old.y-a[-1].y
        assert new.x-b[-1].x==pytest.approx(cos(1.)*dx-sin(1.)*dy)
        assert new.y-b[-1].y==pytest.approx(sin(1.)*dx+cos(1.)*dy)


@pytest.mark.parametrize('kind',['frame','child','time','receipt','jump','turn','gap'])
def test_source_discontinuity_never_borrows_old_trace(kind):
    trail=straight();current=sample(101.3,.325);child='base'
    if kind=='frame':current=sample(101.3,.325,frame='new_odom')
    elif kind=='child':child='new_base'
    elif kind=='time':current=(101.3,99_000_000_000,.325,0.,0.,'odom')
    elif kind=='receipt':current=(99.,101_300_000_000,.325,0.,0.,'odom')
    elif kind=='jump':current=sample(101.3,1.)
    elif kind=='turn':current=sample(101.3,.325,yaw=.5)
    elif kind=='gap':current=sample(103.,.325)
    assert trail.observe(current,child)
    assert len(trail.points)==1


def test_duplicate_receipt_does_not_rejuvenate_source():
    trail=straight();receipt=trail.latest.received_at
    assert trail.observe((101.7,101_200_000_000,.3,0.,0.,'odom'),'base')
    assert trail.latest.received_at==receipt
    assert not trail.project((.3,0.,0.),sample(101.2,.3),now=102.)


def test_changed_duplicate_stamp_invalidates_trace():
    trail=straight()
    assert not trail.observe((101.3,101_200_000_000,.31,0.,0.,'odom'),'base')
    assert not trail.points and trail.latest is None


def test_stationary_two_mm_noise_does_not_hide_actual_rear_corridor():
    trail=straight();trail.observe(sample(101.3,.2985),'base')
    projected=trail.project((.2985,0.,0.),sample(101.3,.2985),now=101.3)
    assert projected[-1].x==pytest.approx(.2985)
    assert projected[-2].x < projected[-1].x-.01


def test_frame_pair_and_receipt_freshness_are_required():
    trail=straight()
    assert not trail.project((.3,0.,0.),sample(101.2,.3,frame='other'),now=101.2)
    assert not trail.project((.3,0.,0.),sample(101.2,.3),now=100.)
    assert not trail.project((.3,0.,0.),sample(101.2,.3),now=102.)
