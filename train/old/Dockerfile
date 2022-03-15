FROM julia:latest


WORKDIR /Plasma
ADD . .

ENV JULIA_PROJECT=/Plasma

RUN julia -e 'using Pkg; Pkg.instantiate()'
RUN julia -e 'using Pkg; Pkg.status()'

# Precompile Plasma.jl
RUN julia -e "using Plasma"
