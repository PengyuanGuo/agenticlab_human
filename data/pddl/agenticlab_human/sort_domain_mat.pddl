(define (domain vlmrobobench)
    (:requirements :strips :typing)
    (:types 
        movable container
        fruit cube - movable
        bowl cardboard_box - container)
    (:predicates 
          (on-mat ?x - movable)
          (hand-empty)
          (holding ?x - movable)
          (in ?x - movable ?y - container))

    (:action pick
    :parameters (?x - movable)
    :precondition (and (hand-empty) 
                      (on-mat ?x))                      
    :effect (and (holding ?x)
                (not (hand-empty))
                (not (on-mat ?x)))
    )

    (:action place
    :parameters (?x - movable ?c - container)
    :precondition (holding ?x)              
    :effect (and (hand-empty)
                (in ?x ?c)
                (not (holding ?x)))
    )
)
