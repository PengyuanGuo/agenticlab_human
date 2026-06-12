(define (domain vlmrobobench)
  (:requirements :strips :typing)
  (:types
    cube cloth plate bowl mat paper - object
    purple-cube blue-cube orange-cube yellow-cube green-cube - cube
    pink-plate cream-plate cyan-plate - plate
  )
  (:predicates
    (on-top-of ?o1 - object ?o2 - object)
    (hand-empty)
    (holding ?o - object)
    (clear ?o - object)
  )

  (:action pick
    :parameters (?obj - object ?underobj - object)
    :precondition (and
      (hand-empty)
      (clear ?obj)
      (on-top-of ?obj ?underobj)
    )
    :effect (and
      (holding ?obj)
      (clear ?underobj)
      (not (hand-empty))
      (not (on-top-of ?obj ?underobj))
      (not (clear ?obj))
    )
  )

  (:action place
    :parameters (?obj - object ?target - object)
    :precondition (and
      (holding ?obj)
      (clear ?target)
    )
    :effect (and
      (on-top-of ?obj ?target)
      (hand-empty)
      (clear ?obj)
      (not (clear ?target))
      (not (holding ?obj))
    )
  )
)