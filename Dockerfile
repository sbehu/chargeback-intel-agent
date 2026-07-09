# Step 1: Use an official, lightweight Python base image
FROM python:3.11-slim

# Step 2: Set the working directory inside the virtual container
WORKDIR /app

# Step 3: Install 'uv' inside the container to handle our dependencies fast
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Step 4: Copy our environment configuration and lockfiles first
COPY pyproject.toml uv.lock ./

# Step 5: Install all required Python packages using uv
RUN uv sync --frozen --no-cache

# Step 6: Copy the rest of your agent application code into the container
COPY . /app

# Step 7: Tell the container exactly what command to run when it boots up on AWS
CMD ["uv", "run", "orchestrator.py"]