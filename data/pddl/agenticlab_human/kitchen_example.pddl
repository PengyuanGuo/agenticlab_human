(define (problem vlmrobobench-potlid-chicken-spice)
    (:domain vlmrobobench)

    (:objects
        the-table - container
        blue-mat-1 - blue-mat
        blue-plate-1 - blue-plate
        bowl-1 - bowl
        pot-1 - pot
        top-drawer-1 - drawer
        pot-lid-1 - pot-lid
        chicken-leg-1 - chicken-leg
        spice-bottle-1 - spice-bottle
    )

    (:init
        (hand-empty)
        (closed pot-1)               ; The pot is closed
        (at pot-lid-1 pot-1)         ; The lid is ON the pot

        ; Other open containers
        (opened the-table)
        (opened blue-mat-1)
        (opened blue-plate-1)
        (opened bowl-1)
        
        ; Drawer state
        (closed top-drawer-1)

        ; Item locations
        (at chicken-leg-1 bowl-1)
        (at spice-bottle-1 top-drawer-1)

        ; Lid relation
        (is-lid-of pot-lid-1 pot-1)
    )

    (:goal
        (and
            (at pot-lid-1 blue-mat-1)
            (at chicken-leg-1 pot-1)
            (at spice-bottle-1 blue-plate-1)
        )
    )
)