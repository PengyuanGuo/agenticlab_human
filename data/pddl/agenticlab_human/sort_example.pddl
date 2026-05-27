(define (problem sort-fruit-cube)
  (:domain vlmrobobench)

  (:objects
    apple - fruit
    yellow-cube - cube
    bowl1 - bowl
    box1 - cardboard_box
  )

  (:init
    (on-table apple)
    (on-table yellow-cube)
    (hand-empty)
  )

  (:goal
    (and
      (in apple bowl1)
      (in yellow-cube box1)
    )
  )
)
