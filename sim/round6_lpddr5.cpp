// Read-only x16 BL16 FIFO timing model. Values from the pinned
// Ramulator 2.1 LPDDR5_6400 BG preset (see Round 6 protocol).
#include <algorithm>
#include <array>
#include <cstdint>

extern "C" void simulate(const int64_t* starts, const int64_t* burst_counts,
                         const int64_t* bounds, const int64_t* before,
                         int64_t descriptor_clocks, int64_t* output) {
    constexpr int BANKS=16;
    constexpr int ROW_BURSTS=64;
    constexpr int64_t NEG=-(1LL<<40);
    constexpr int tRCD=15,tRP=15,tRAS=34,tRC=49,tRTP=6,tCL=17,tBL=2,
                  tCCD_S=2,tCCD_L=4,tRRD=4,tFAW=16;
    std::array<int64_t,BANKS> open, act, read;
    open.fill(-1); act.fill(NEG); read.fill(NEG);
    std::array<int64_t,4> issued;
    issued.fill(NEG);
    std::array<int64_t,4> bg_rd;
    bg_rd.fill(NEG);
    std::array<int64_t,32> completions;
    completions.fill(NEG);
    int64_t requests=0, activates=0, hits=0, precharges=0;
    int64_t ca=0, last_read=NEG,last_act=NEG, finish=0;
    int64_t act_count=0;
    for (int stage=0;stage<4;++stage) {
        const int64_t ready=finish+before[stage];
        int64_t issue=ready;
        for (int64_t i=bounds[stage];i<bounds[stage+1];++i) {
            issue+=descriptor_clocks;
            for (int64_t j=0;j<burst_counts[i];++j) {
                const int64_t addr=starts[i]+j;
                const int64_t stripe=addr/ROW_BURSTS;
                const int group=stripe%4;
                const int bank=stripe%BANKS;
                const int64_t row=stripe/BANKS;
                const int64_t arrival=std::max(issue,completions[requests%32]);
                int64_t start=std::max(ca,arrival);
                if(open[bank]!=row) {
                    if(open[bank]>=0) {
                        start=std::max({start,act[bank]+tRAS,read[bank]+tRTP});
                        start+=tRP;
                        ++precharges;
                    }
                    int64_t a=std::max({start,last_act+tRRD,issued[act_count%4]+tFAW,act[bank]+tRC});
                    // ACT1 and ACT2 consume separate CA command slots.
                    act[bank]=a+1;
                    ca=act[bank]+1;
                    last_act=act[bank];
                    issued[act_count%4]=act[bank];
                    ++act_count;
                    open[bank]=row;
                    ++activates;
                    start=std::max(ca,act[bank]+tRCD);
                } else {
                    ++hits;
                }
                int64_t rd=std::max({start,last_read+tCCD_S,bg_rd[group]+tCCD_L});
                ca=rd+1;
                read[bank]=rd; last_read=rd; bg_rd[group]=rd;
                finish=rd+tCL+tBL;
                completions[requests%32]=finish;
                ++requests;
            }
        }
        finish=std::max(finish,ready);
        output[stage]=finish;
    }
    output[4]=requests; output[5]=hits; output[6]=activates; output[7]=precharges;
}
