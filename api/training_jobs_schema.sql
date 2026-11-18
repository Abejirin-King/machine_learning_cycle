-- Run this SQL in the Supabase SQL Editor before enabling remote retraining.
create table if not exists public.training_workers (
  id uuid primary key,
  name text not null,
  token_hash text not null,
  enabled boolean not null default true,
  last_seen_at timestamptz,
  current_job_id uuid,
  created_at timestamptz not null default now()
);

create table if not exists public.training_jobs (
  id uuid primary key,
  status text not null check (status in ('queued','running','candidate_ready','completed','failed')),
  epochs integer not null check (epochs between 1 and 5),
  metadata_path text not null,
  audio_paths jsonb not null,
  recording_count integer not null,
  worker_id uuid references public.training_workers(id),
  result jsonb,
  candidate_path text,
  error text,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz
);

create index if not exists training_jobs_status_created_idx
  on public.training_jobs(status, created_at);

create or replace function public.claim_training_job(p_worker_id uuid)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  claimed public.training_jobs;
begin
  update public.training_workers
     set last_seen_at = now()
   where id = p_worker_id and enabled = true;
  if not found then return null; end if;

  select * into claimed
    from public.training_jobs
   where status = 'queued'
   order by created_at
   for update skip locked
   limit 1;

  if not found then return null; end if;

  update public.training_jobs
     set status = 'running', worker_id = p_worker_id, started_at = now()
   where id = claimed.id
   returning * into claimed;

  update public.training_workers
     set current_job_id = claimed.id
   where id = p_worker_id;

  return jsonb_build_object(
    'id', claimed.id, 'epochs', claimed.epochs,
    'metadata_path', claimed.metadata_path,
    'audio_paths', claimed.audio_paths,
    'recording_count', claimed.recording_count
  );
end;
$$;

-- Keep tables inaccessible to browser clients. The FastAPI service uses the service-role key.
alter table public.training_workers enable row level security;
alter table public.training_jobs enable row level security;

-- Only the FastAPI service-role client may claim jobs.
revoke all on function public.claim_training_job(uuid) from public;
grant execute on function public.claim_training_job(uuid) to service_role;

-- Private object storage for uploaded audio and candidate checkpoints.
insert into storage.buckets (id, name, public, file_size_limit)
values ('training-jobs', 'training-jobs', false, 41943040)
on conflict (id) do update set public = false, file_size_limit = 41943040;
