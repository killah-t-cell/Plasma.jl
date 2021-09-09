FROM julia:latest

CMD ["julia", "/Plasma/src/Plasma.jl"]

WORKDIR /Plasma
ADD . .

ENV JULIA_NUM_PRECOMPILE_TASKS=1 

COPY Manifest.toml /Plasma/Manifest.toml
COPY Project.toml /Plasma/Project.toml

ENV JULIA_PROJECT=/Plasma

RUN julia -e 'using Pkg; Pkg.instantiate()'
RUN julia -e 'using Pkg; Pkg.status()'

# Precompile Plasma.jl
RUN julia -e "using Plasma"
