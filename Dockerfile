FROM julia:1.6.1

WORKDIR /Plasma

COPY . .

COPY Project.toml /Plasma/Project.toml

ENV JULIA_PROJECT=/Plasma
# ENV JULIA_NUM_PRECOMPILE_TASKS=1

RUN julia -e 'using Pkg; Pkg.instantiate()'
# RUN julia -e 'using Pkg; Pkg.status()'

# Precompile Plasma.jl
RUN julia -e "using Plasma"

# install Python and Jupyter (required for paperspace)
# RUN apt-get update && apt-get install -y python3-pip python3-dev && pip3 install jupyter

CMD ["julia", "/Plasma/src/Plasma.jl"]
