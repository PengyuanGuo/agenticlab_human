(define (domain vlmrobobench)
    (:requirements :strips :typing)
    (:types block cloth plate bowl apple cup - object
            wooden-block blue-block orange-block yellow-block green-block - block)
    (:predicates 
          (on-table ?o - object)
          (on-top-of ?o1 - object ?o2 - object)
          (hand-empty)
          (holding ?o - object)
          (clear ?o - object))

    (:action pick
    :parameters (?obj - object)
    :precondition (and (hand-empty) 
                      (clear ?obj)
                      (on-table ?obj))                      
    :effect (and (holding ?obj)
                (not (hand-empty))
                (not (on-table ?obj))
                (not (clear ?obj))))
        

    (:action place
    :parameters (?obj - object ?target - object)
    :precondition (and (holding ?obj) 
                      (clear ?target))               
    :effect (and (on-top-of ?obj ?target)
                (hand-empty)
                (clear ?obj)
                (not (clear ?target))
                (not (holding ?obj)))))
