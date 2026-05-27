(define (problem stack-cubes-tower)
  (:domain vlmrobobench)

  (:objects
    orange-cube - cube
    yellow-cube - cube
    blue-cube - cube
    plate - plate
    blue-mat - mat
  )

  (:init
    (on-top-of orange-cube plate)
    (on-top-of yellow-cube orange-cube)
    (on-top-of blue-cube yellow-cube)
    (clear blue-mat)
    (clear blue-cube)
    (hand-empty)
  )

  (:goal
    (and
      (on-top-of blue-cube blue-mat)
      (on-top-of yellow-cube blue-cube)
      (on-top-of orange-cube yellow-cube)
    )
  )
)