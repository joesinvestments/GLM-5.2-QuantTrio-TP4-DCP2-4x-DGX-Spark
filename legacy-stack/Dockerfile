FROM vllm-node-tf5-glm52-b12x:probe
# Legacy GLM stack challenger: ciprianveg/CosmicRaisins mods + tonyd2wild overhang patch.
COPY kernels/ /root/models/models15/glm-triton/
COPY mods/ /tmp/mods/
COPY patches/ /tmp/patches/
RUN bash /tmp/mods/glm52-sm12x-sparse/run.sh && \
    bash /tmp/mods/glm52-b12x-sparse/run.sh && \
    python3 /tmp/patches/fix-indexer-mtp-overhang.py && \
    rm -rf /tmp/mods /tmp/patches
