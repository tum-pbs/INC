#include "PISO_solver.h"


index_t PISOstep(std::shared_ptr<Domain> domain, const index_t steps, const index_t correctorSteps, const torch::Tensor &timeStep, const bool advect){

    const bool prev_as_initial_guess_advection = true;
    const bool prev_as_initial_guess_pressure = true;

    for(index_t step=0; step<steps; ++step){
        if(!advect){
            CopyVelocityResultFromBlocks(domain);
            for(auto block : domain->blocks){
                block->velocity.zero_();
                block->pressure.zero_();
            }
        }
    }
}