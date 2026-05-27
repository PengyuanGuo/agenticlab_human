(define (problem vlmrobobench-stack-cubes-on-pink-plate)
  (:domain vlmrobobench)

  (:objects
    orange-cube-1 - orange-cube
    yellow-cube-1 - yellow-cube
    green-cube-1 - green-cube
    blue-cube-1 - blue-cube
    pink-plate-1 - pink-plate
    cream-plate-1 - cream-plate
    cyan-plate-1 - cyan-plate
  )

  (:init
    (on-top-of blue-cube-1 cream-plate-1)
    (on-top-of yellow-cube-1 blue-cube-1)
    (on-top-of orange-cube-1 cyan-plate-1)
    (on-top-of green-cube-1 orange-cube-1)
    (clear yellow-cube-1)
    (clear green-cube-1)
    (clear pink-plate-1)
    (hand-empty)
  )

  (:goal
    (and
      (on-top-of orange-cube-1 pink-plate-1)
      (on-top-of yellow-cube-1 orange-cube-1)
      (on-top-of green-cube-1 yellow-cube-1)
      (on-top-of blue-cube-1 green-cube-1)
    )
  )
)