(define (domain vlmrobobench)
    (:requirements :strips :typing)
    
    (:types
        item - object
        container - object
        drawer - container
        blue-mat container
        lid - item
        pot bowl blue-plate - container
        spice-bottle chicken-leg - item
        pot-lid - lid
    )
    
    (:predicates
        (hand-empty)                          ; robot's gripper is empty
        (holding ?x - item)                   ; robot is holding item x
        (opened ?d - container)                  ; Container is accessible
        (closed ?d - container)                  ; Container is inaccessible
        (at ?x - item ?c - container)     ; item x is inside drawer d
        (on-table ?x - item)                  ; item x is on the table
        (is-lid-of ?l - lid ?c - container)   ; Defines which lid fits which pot
    )
    
    (:action open-drawer
        :parameters (?d - drawer)
        :precondition (and (hand-empty) (closed ?d))
        :effect (and (opened ?d) (not (closed ?d)))
    )
    
    (:action close-drawer
        :parameters (?d - drawer)
        :precondition (and (hand-empty) (opened ?d))
        :effect (and (closed ?d) (not (opened ?d)))
    )
    
    (:action uncap-container
        :parameters (?l - lid ?c - container)
        :precondition (and (hand-empty)(at ?l ?c)(is-lid-of ?l ?c)(closed ?c))
        :effect (and (holding ?l)(not (hand-empty))(not (at ?l ?c))(opened ?c)(not (closed ?c)))
        )
    
    (:action cap
        :parameters (?l - lid ?c - container)
        :precondition (and (holding ?l)(opened ?c)(is-lid-of ?l ?c))
        :effect (and (hand-empty)(not (holding ?l))(at ?l ?c)(closed ?c)(not (opened ?c)))
    )

    (:action pick
        :parameters (?x - item ?c - container)
        :precondition (and 
            (hand-empty)
            (at ?x ?c)      ; Item must be at this location
            (opened ?c)     ; The location must be open (Table is always open)
        )
        :effect (and 
            (holding ?x)
            (not (hand-empty))
            (not (at ?x ?c))
        )
    )
    
    (:action place
        :parameters (?x - item ?c - container)
        :precondition (and 
            (holding ?x)
            (opened ?c)     ; Critical: Can't place in closed drawer, but Table is always open
            (not (is-lid-of ?x ?c)) ; You must use 'cap' (to close it) or place it elsewhere.
        )
        :effect (and 
            (hand-empty)
            (at ?x ?c)
            (not (holding ?x))
        ))
)
