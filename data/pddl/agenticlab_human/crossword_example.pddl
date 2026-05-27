(define (problem crossword-puzzle)
  (:domain vlmrobobench)

  (:objects
    block-O - letter-block
    block-E - letter-block
    slot-1 - number-slot
    slot-2 - number-slot
    slot-3 - number-slot
  )

  (:init
    (on-table block-O)
    (on-table block-E)
    (hand-empty)
  )

  (:goal
    (and
      (in block-E slot-2)
    )
  )
)