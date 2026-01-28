

# run sensor (2+i), where i the ith instrument, and the last instrument is doing nothing
# declare dead (second to last) declare alive (last action)
POMDPs.actions(pomdp::LifeDetectionPOMDP) = [1:pomdp.inst..., pomdp.inst+1, pomdp.inst+2]

POMDPs.actionindex(pomdp::LifeDetectionPOMDP, a::Int) = a
