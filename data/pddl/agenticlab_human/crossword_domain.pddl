(define (domain vlmrobobench)
    (:requirements :strips :typing)
    (:types 
        block - movable
        letter-block - block
        slot - unmovable
        number-slot - slot)
    (:predicates 
          (on-table ?x - movable) ; The block is on the table surface
          (hand-empty)            ; The robot's gripper is empty
          (holding ?x - movable)  ; The robot is holding a block
          (in ?x - movable ?y - unmovable)) ; The block is placed inside a slot
    (:action pick
    :parameters (?x - movable)
    :precondition (and (hand-empty) 
                      (on-table ?x))                      
    :effect (and (holding ?x)
                (not (hand-empty))
                (not (on-table ?x)))
    )
    (:action place
    :parameters (?x - movable ?c - unmovable)
    :precondition (holding ?x)              
    :effect (and (hand-empty)
                (in ?x ?c)
                (not (holding ?x)))
    )
)