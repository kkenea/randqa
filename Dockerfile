FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Project files
COPY pyproject.toml README.md ./
COPY src ./src

# Install project with ALL extras (web, gui, ai) + gunicorn
RUN pip install --upgrade pip wheel setuptools \
 && pip install ".[web,gui,ai]" gunicorn>=21

# Change ownership to appuser for proper permissions
RUN chown -R appuser:appuser /app /home/appuser

# Expose API and GUI ports
EXPOSE 8000 8501

USER appuser

# Default message; compose overrides with the right command
CMD ["python", "-c", "print('Use docker compose to run: api or gui')"]
