(define (domain vlmrobobench)
    (:requirements :strips :typing)
    (:types 
        movable container
        fruit cube - movable
        bowl cardboard_box - container)
    (:predicates 
          (on-table ?x - movable)
          (hand-empty)
          (holding ?x - movable)
          (in ?x - movable ?y - container))

    (:action pick
    :parameters (?x - movable)
    :precondition (and (hand-empty) 
                      (on-table ?x))                      
    :effect (and (holding ?x)
                (not (hand-empty))
                (not (on-table ?x)))
    )

    (:action place
    :parameters (?x - movable ?c - container)
    :precondition (holding ?x)              
    :effect (and (hand-empty)
                (in ?x ?c)
                (not (holding ?x)))
    )
)
