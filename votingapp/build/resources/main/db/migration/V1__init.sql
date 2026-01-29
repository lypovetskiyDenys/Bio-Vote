CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255),
    face_embedding BYTEA NOT NULL
);

CREATE TABLE polls (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    end_date DATE NOT NULL,
    author_id BIGINT,
    FOREIGN KEY (author_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE votes (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    voted_at TIMESTAMP NOT NULL,
    poll_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    FOREIGN KEY (poll_id) REFERENCES polls (id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT unique_user_poll UNIQUE (user_email, poll_id)
);
