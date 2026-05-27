(define (problem vlmrobobench-example)
(:domain vlmrobobench)
(:objects
blue-cube-1 - blue-cube
wooden-cube-1 - wooden-cube)
(:init
(on-table blue-cube-1)
(on-table wooden-cube-1)
(clear blue-cube-1)
(clear wooden-cube-1)
(hand-empty)
)
(:goal
(and
(on-top-of wooden-cube-1 blue-cube-1)
)
)
)
