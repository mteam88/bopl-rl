#include "bopl.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *state(BoplWorld *world) {
    size_t size=0;
    assert(bopl_state_json(world,NULL,0,&size)==BOPL_BUFFER_TOO_SMALL);
    char *json=malloc(size); assert(json);
    assert(bopl_state_json(world,json,size,&size)==BOPL_OK);
    return json;
}
int main(void) {
    _Static_assert(sizeof(BoplAction)==2,"action layout");
    _Static_assert(sizeof(BoplObservation)==80,"observation layout");
    assert(bopl_abi_version()==1);
    BoplWorld *a=bopl_create(), *b=bopl_create(); assert(a && b);
    char *initial=state(a);
    BoplAction inputs[2]={{1,0},{128,0}};
    for(int tick=0;tick<100;tick++) {
        assert(bopl_step(a,inputs,2)==BOPL_OK);
        assert(bopl_step(b,inputs,2)==BOPL_OK);
    }
    char *left=state(a),*right=state(b);assert(strcmp(left,right)==0);
    inputs[0].buttons=128;
    assert(bopl_step(a,inputs,2)==BOPL_UNSUPPORTED);
    assert(bopl_step(a,inputs,1)==BOPL_INVALID_ARGUMENT);
    char *unchanged=state(a);assert(strcmp(left,unchanged)==0);
    BoplObservation observations[2];assert(bopl_observe(a,observations,2)==BOPL_OK);
    assert(observations[0].id==1 && observations[1].id==2);
    assert(bopl_reset(a)==BOPL_OK);
    char *reset=state(a);assert(strcmp(initial,reset)==0);
    assert(bopl_terminal(NULL)==-1);
    bopl_destroy(a);bopl_destroy(b);bopl_destroy(NULL);
    free(initial);free(left);free(right);free(unchanged);free(reset);
    assert(bopl_create_players(0)==NULL && bopl_create_players(7)==NULL);
    assert(bopl_player_count(NULL)==0);
    for(size_t count=1;count<=6;count++) {
        BoplWorld *world=bopl_create_players(count);assert(world);
        assert(bopl_player_count(world)==count);
        BoplAction actions[6]={{0}};
        BoplObservation values[6];
        char *before=state(world);
        assert(bopl_step(world,actions,count-1)==BOPL_INVALID_ARGUMENT);
        assert(bopl_observe(world,values,count-1)==BOPL_INVALID_ARGUMENT);
        assert(bopl_step(world,actions,count)==BOPL_OK);
        assert(bopl_observe(world,values,count)==BOPL_OK);
        for(size_t p=0;p<count;p++) assert(values[p].id==p+1 && values[p].alive);
        assert(bopl_reset(world)==BOPL_OK);
        char *after=state(world);assert(strcmp(before,after)==0);
        free(before);free(after);bopl_destroy(world);
    }
    puts("C_ABI_SMOKE_OK");
}
